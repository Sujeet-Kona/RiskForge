"""The scoring service: one path from features to decision, at full precision.

Requirement 6 is the point of this file. A probability that sits just under a
threshold must be decided on its own digits, not on a rounded copy of them, so
the calibrator is replaced with one that returns values engineered to straddle
``T_REVIEW`` and ``T_BLOCK``.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from app.schemas import RiskScoreRequest
from app.service import RiskScoringService, to_response
from explain.model_explanation import (
    ATTRIBUTION_SPACE,
    RISK_DECREASING,
    RISK_INCREASING,
)
from ml.models.artifact import ArtifactError, RiskForgeArtifact
from ml.models.calibration import FraudCalibrator
from policy.decision import Decision


class _ConstantCalibrator:
    """Stands in for the fitted isotonic mapping and returns a fixed score."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(scores, dtype=float), self.value)


def _service_with_probability(artifact: RiskForgeArtifact, value: float):
    """A scoring service whose calibrated probability is exactly ``value``."""
    doctored = dataclasses.replace(
        artifact,
        calibrator=FraudCalibrator(method="isotonic", estimator=_ConstantCalibrator(value)),
    )

    return RiskScoringService.from_artifact(doctored)


@pytest.fixture
def minimal_request() -> RiskScoreRequest:
    """A transaction with no history and no optional fields at all."""
    return RiskScoreRequest(
        transaction={
            "TransactionID": "1",
            "TransactionDT": 1_000,
            "TransactionAmt": 10.0,
            "card1": 1,
        }
    )


def test_a_scored_transaction_returns_a_complete_assessment(serving_service, score_payload):
    assessment = serving_service.score(RiskScoreRequest(**score_payload))

    assert assessment.transaction_id == score_payload["transaction"]["TransactionID"]
    assert assessment.model_version == serving_service.artifact.model_version
    assert assessment.feature_version == serving_service.artifact.feature_version
    assert assessment.history_transactions_used == len(score_payload["history"])


def test_the_probability_is_inside_the_unit_interval(serving_service, score_payload):
    probability = serving_service.score(RiskScoreRequest(**score_payload)).risk_probability

    assert 0.0 <= probability <= 1.0


def test_the_decision_is_one_of_the_three_locked_bands(serving_service, score_payload):
    assessment = serving_service.score(RiskScoreRequest(**score_payload))

    assert assessment.decision in {Decision.ALLOW, Decision.REVIEW, Decision.BLOCK}


def test_history_length_is_reported_not_inferred(serving_service, score_payload):
    payload = copy.deepcopy(score_payload)
    payload["history"] = payload["history"][:1]

    assessment = serving_service.score(RiskScoreRequest(**payload))

    assert assessment.history_transactions_used == 1


def test_an_empty_history_is_scored_without_error(serving_service, score_payload):
    payload = copy.deepcopy(score_payload)
    payload["history"] = []

    assert serving_service.score(RiskScoreRequest(**payload)).risk_probability >= 0.0


# --- requirement 6: the policy sees the unrounded probability -----------------


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0749999, Decision.ALLOW),
        (0.0750001, Decision.REVIEW),
        (0.075, Decision.REVIEW),
        (0.1099999, Decision.REVIEW),
        (0.1100001, Decision.BLOCK),
        (0.110, Decision.BLOCK),
        (0.0, Decision.ALLOW),
        (1.0, Decision.BLOCK),
    ],
)
def test_the_band_follows_the_full_precision_probability(
    loaded_artifact, minimal_request, probability, expected
):
    service = _service_with_probability(loaded_artifact, probability)
    request = minimal_request

    assert service.score(request).decision is expected


@pytest.mark.parametrize("probability", [0.0749999, 0.0750001])
def test_a_value_that_rounds_to_the_threshold_is_not_decided_on_its_rounding(
    loaded_artifact, minimal_request, probability
):
    """Both round to 0.075 at three decimals, and they must land in different bands."""
    assert round(probability, 3) == 0.075

    service = _service_with_probability(loaded_artifact, probability)
    request = minimal_request
    assessment = service.score(request)

    assert assessment.risk_probability == probability
    assert assessment.decision is (
        Decision.ALLOW if probability < 0.075 else Decision.REVIEW
    )


@pytest.mark.parametrize("probability", [0.1099999, 0.1100001])
def test_a_value_that_rounds_to_the_block_threshold_is_not_decided_on_its_rounding(
    loaded_artifact, minimal_request, probability
):
    assert round(probability, 3) == 0.11

    service = _service_with_probability(loaded_artifact, probability)
    request = minimal_request
    assessment = service.score(request)

    assert assessment.risk_probability == probability
    assert assessment.decision is (
        Decision.REVIEW if probability < 0.110 else Decision.BLOCK
    )


# --- requirement 24: determinism ---------------------------------------------


def test_scoring_the_same_request_twice_gives_the_same_result(
    serving_service, score_payload
):
    request = RiskScoreRequest(**score_payload)

    first = serving_service.score(request)
    second = serving_service.score(request)

    assert first.risk_probability == second.risk_probability
    assert first.decision is second.decision
    assert first.explanation.model_explanation.top_reason_codes == (
        second.explanation.model_explanation.top_reason_codes
    )
    assert first.explanation.decision_explanation.explanation == (
        second.explanation.decision_explanation.explanation
    )


def test_the_reported_probability_is_the_one_the_explanation_was_given(
    serving_service, score_payload
):
    assessment = serving_service.score(RiskScoreRequest(**score_payload))

    assert (
        assessment.explanation.calibrated_risk_probability == assessment.risk_probability
    )


# --- the service reuses the artifact instead of rebuilding it ----------------


def test_the_model_view_wraps_the_artifact_pipeline_without_copying_it(serving_service):
    assert serving_service.model.model is serving_service.artifact.pipeline


def test_the_model_view_reports_the_artifact_feature_contract(serving_service):
    assert serving_service.model.feature_columns == list(
        serving_service.artifact.feature_columns
    )
    assert serving_service.model.scale_pos_weight == serving_service.artifact.scale_pos_weight


def test_the_service_uses_the_frozen_policy_thresholds(serving_service):
    assert serving_service.policy.config.policy_version == (
        serving_service.artifact.policy_version
    )
    assert serving_service.policy.config.review_threshold == (
        serving_service.artifact.review_threshold
    )
    assert serving_service.policy.config.block_threshold == (
        serving_service.artifact.block_threshold
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy_version", "policy-002"),
        ("review_threshold", 0.05),
        ("block_threshold", 0.20),
    ],
)
def test_an_artifact_built_under_a_different_policy_is_refused_at_startup(
    loaded_artifact, field, value
):
    """Serving decisions that contradict their own explanation is worse than failing."""
    doctored = dataclasses.replace(loaded_artifact, **{field: value})

    with pytest.raises(ArtifactError):
        RiskScoringService.from_artifact(doctored)


# --- to_response --------------------------------------------------------------


def test_the_response_carries_every_field_the_brief_requires(serving_service, score_payload):
    assessment = serving_service.score(RiskScoreRequest(**score_payload))
    response = to_response(assessment)

    assert response.risk_probability == assessment.risk_probability
    assert response.decision == assessment.decision.value
    assert response.model_version == "xgb-001"
    assert response.feature_version == "behavioral-v1"
    assert response.policy_version == "policy-001"
    assert response.feature_count == 28
    assert response.reason_codes != []
    assert response.decision_explanation


def test_the_decision_rule_is_reported_separately_from_the_model_signals(
    serving_service, score_payload
):
    response = to_response(serving_service.score(RiskScoreRequest(**score_payload)))

    model_codes = {entry.code for entry in response.reason_codes}

    assert response.decision_reason_code.code not in model_codes
    assert response.decision_reason_code.code.startswith("POLICY_")


def test_risk_decreasing_signals_are_not_presented_as_fraud_evidence(
    serving_service, score_payload
):
    response = to_response(serving_service.score(RiskScoreRequest(**score_payload)))

    for entry in response.risk_decreasing_signals:
        assert entry.direction == RISK_DECREASING
        assert entry.contribution <= 0.0

    for entry in response.reason_codes:
        assert entry.direction == RISK_INCREASING
        assert entry.contribution >= 0.0


@pytest.mark.parametrize("top_k", [1, 2, 5])
def test_top_k_bounds_the_number_of_model_signals(serving_service, score_payload, top_k):
    payload = copy.deepcopy(score_payload)
    payload["top_k"] = top_k

    response = to_response(serving_service.score(RiskScoreRequest(**payload)))

    assert len(response.reason_codes) <= top_k
    assert len(response.risk_decreasing_signals) <= top_k


def test_the_policy_thresholds_are_echoed_in_the_response(serving_service, score_payload):
    response = to_response(serving_service.score(RiskScoreRequest(**score_payload)))

    assert response.policy_detail.review_threshold == 0.075
    assert response.policy_detail.block_threshold == 0.110


def test_the_attribution_space_is_declared_in_the_response(serving_service, score_payload):
    response = to_response(serving_service.score(RiskScoreRequest(**score_payload)))

    assert response.model_detail.attribution_space == ATTRIBUTION_SPACE
    assert response.model_detail.attribution_space_note
