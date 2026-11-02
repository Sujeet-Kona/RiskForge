from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from explain.decision_explanation import DecisionExplanation
from explain.explanation import (
    TRANSACTION_ID_COLUMN,
    TransactionExplanation,
    explain_transaction,
)
from explain.model_explanation import ModelExplanation
from ml.data.modeling import TARGET_COLUMN
from policy.decision import Decision, DecisionPolicy


LOW_RISK = 0.004
REVIEW_BAND = 0.090
BLOCK_BAND = 0.900

EXPLANATION_FIELDS = {
    "transaction_id",
    "calibrated_risk_probability",
    "decision",
    "decision_explanation",
    "model_explanation",
    "top_reason_codes",
}


@pytest.fixture(scope="module")
def transaction(tiny_frame: pd.DataFrame) -> pd.DataFrame:
    return tiny_frame.iloc[[0]]


def explain(tiny_explainer, transaction, policy, probability) -> TransactionExplanation:
    return explain_transaction(
        tiny_explainer,
        transaction,
        calibrated_risk_probability=probability,
        policy=policy,
    )


# --- the two halves stay separate ---------------------------------------------


def test_an_explanation_carries_both_halves(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    explained = explain(tiny_explainer, transaction, tiny_policy, REVIEW_BAND)

    assert isinstance(explained, TransactionExplanation)
    assert isinstance(explained.model_explanation, ModelExplanation)
    assert isinstance(explained.decision_explanation, DecisionExplanation)


def test_the_decision_half_agrees_with_the_policy_itself(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    for probability, expected in (
        (LOW_RISK, Decision.ALLOW),
        (REVIEW_BAND, Decision.REVIEW),
        (BLOCK_BAND, Decision.BLOCK),
    ):
        explained = explain(tiny_explainer, transaction, tiny_policy, probability)

        assert explained.decision is expected.value
        assert explained.decision_explanation.decision is expected
        assert tiny_policy.decide(probability).decision is expected


def test_the_calibrated_probability_is_passed_through_not_recomputed(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    """The layer never refits calibration, so the caller's number is authoritative."""
    explained = explain(tiny_explainer, transaction, tiny_policy, REVIEW_BAND)

    assert explained.calibrated_risk_probability == REVIEW_BAND
    assert explained.decision_explanation.risk_probability == REVIEW_BAND


def test_the_calibrated_probability_is_not_the_raw_model_probability(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    """Requirement: raw score and calibrated risk are reported as different things."""
    low = explain(tiny_explainer, transaction, tiny_policy, LOW_RISK)
    high = explain(tiny_explainer, transaction, tiny_policy, BLOCK_BAND)

    assert low.calibrated_risk_probability != high.calibrated_risk_probability
    assert (
        low.model_explanation.raw_model_probability
        == high.model_explanation.raw_model_probability
    )
    assert low.model_explanation.raw_margin == high.model_explanation.raw_margin


def test_the_decision_half_carries_no_feature_information(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    payload = explain(
        tiny_explainer, transaction, tiny_policy, REVIEW_BAND
    ).decision_explanation.to_dict()

    assert "feature" not in payload
    assert "features" not in payload
    assert "contribution" not in payload


# --- top reason codes ---------------------------------------------------------


def test_top_reason_codes_append_the_decision_rule_to_the_model_signals(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    explained = explain(tiny_explainer, transaction, tiny_policy, BLOCK_BAND)

    assert explained.top_reason_codes == (
        *explained.model_explanation.top_reason_codes,
        explained.decision_explanation.reason_code,
    )


def test_top_reason_codes_shrink_with_top_k(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    one = explain_transaction(
        tiny_explainer,
        transaction,
        calibrated_risk_probability=BLOCK_BAND,
        policy=tiny_policy,
        top_k=1,
    )
    three = explain_transaction(
        tiny_explainer,
        transaction,
        calibrated_risk_probability=BLOCK_BAND,
        policy=tiny_policy,
        top_k=3,
    )

    assert one.model_explanation.top_k == 1
    assert three.model_explanation.top_k == 3
    assert len(one.top_reason_codes) <= len(three.top_reason_codes)
    assert one.top_reason_codes[-1] == three.top_reason_codes[-1]


def test_the_reason_codes_are_stable_strings(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    explained = explain(tiny_explainer, transaction, tiny_policy, REVIEW_BAND)

    assert explained.top_reason_codes
    assert all(isinstance(code, str) and code for code in explained.top_reason_codes)
    assert all(code == code.upper() for code in explained.top_reason_codes)


# --- transaction identity -----------------------------------------------------


def test_the_transaction_id_is_reported_when_present(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    explained = explain(tiny_explainer, transaction, tiny_policy, LOW_RISK)

    assert explained.transaction_id == transaction[TRANSACTION_ID_COLUMN].iloc[0]


def test_the_transaction_id_is_none_when_absent(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    anonymous = transaction.drop(columns=[TRANSACTION_ID_COLUMN])

    assert (
        explain(tiny_explainer, anonymous, tiny_policy, LOW_RISK).transaction_id is None
    )


# --- determinism, serialisation, validation ------------------------------------


def test_the_same_input_produces_the_same_explanation(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    first = explain(tiny_explainer, transaction, tiny_policy, REVIEW_BAND).to_dict()
    second = explain(tiny_explainer, transaction, tiny_policy, REVIEW_BAND).to_dict()

    assert first == second


def test_the_serialised_form_exposes_both_halves(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    payload = explain(
        tiny_explainer, transaction, tiny_policy, REVIEW_BAND
    ).to_dict()

    assert set(payload) == EXPLANATION_FIELDS
    assert payload["decision"] == Decision.REVIEW.value
    assert payload["calibrated_risk_probability"] == REVIEW_BAND
    assert isinstance(payload["model_explanation"], dict)
    assert isinstance(payload["decision_explanation"], dict)


def test_the_explanation_is_frozen(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    explained = explain(tiny_explainer, transaction, tiny_policy, LOW_RISK)

    with pytest.raises(FrozenInstanceError):
        explained.decision = "BLOCK"  # type: ignore[misc]


def test_a_non_dataframe_is_rejected(tiny_explainer, tiny_policy: DecisionPolicy):
    with pytest.raises(TypeError, match="single-row pandas DataFrame"):
        explain_transaction(
            tiny_explainer,
            {"TransactionAmt": 1.0},  # type: ignore[arg-type]
            calibrated_risk_probability=LOW_RISK,
            policy=tiny_policy,
        )


def test_a_label_column_on_the_row_changes_nothing(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    """The explanation is a function of the features, never of the outcome."""
    fraud = transaction.assign(**{TARGET_COLUMN: 1})
    legitimate = transaction.assign(**{TARGET_COLUMN: 0})

    assert (
        explain(tiny_explainer, fraud, tiny_policy, REVIEW_BAND).to_dict()
        == explain(tiny_explainer, legitimate, tiny_policy, REVIEW_BAND).to_dict()
        == explain(tiny_explainer, transaction, tiny_policy, REVIEW_BAND).to_dict()
    )


def test_two_rows_are_rejected(
    tiny_explainer, tiny_frame: pd.DataFrame, tiny_policy: DecisionPolicy
):
    with pytest.raises(ValueError, match="exactly one transaction row"):
        explain_transaction(
            tiny_explainer,
            tiny_frame.iloc[[0, 1]],
            calibrated_risk_probability=LOW_RISK,
            policy=tiny_policy,
        )


def test_an_out_of_range_probability_is_rejected(
    tiny_explainer, transaction: pd.DataFrame, tiny_policy: DecisionPolicy
):
    with pytest.raises(ValueError):
        explain_transaction(
            tiny_explainer,
            transaction,
            calibrated_risk_probability=1.5,
            policy=tiny_policy,
        )
