"""The one inference path: features, model, calibration, policy, explanation.

Every step delegates to the function the offline pipeline already uses. Nothing
here re-implements scoring, re-derives calibration, restates a threshold or
builds a second explanation format, so an API result and an offline result for
the same transaction are the same computation rather than two computations that
are merely expected to agree.

The service is constructed once at application startup and is read-only for the
lifetime of the process. No method fits anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.features import build_serving_features
from app.schemas import (
    DecisionReasonCodeOut,
    ModelDetailOut,
    PolicyDetailOut,
    ReasonCodeOut,
    RiskScoreRequest,
    RiskScoreResponse,
)
from explain.explanation import TransactionExplanation, explain_transaction
from explain.model_explanation import (
    ATTRIBUTION_SPACE_NOTE,
    FeatureContribution,
    ModelExplainer,
    build_model_explainer,
)
from explain.reason_codes import decision_reason_code, reason_code_for_feature
from ml.data.modeling import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES
from ml.models.artifact import ArtifactError, RiskForgeArtifact
from ml.models.calibration import apply_calibrator
from ml.models.xgboost_model import XGBoostFraudModel, predict_fraud_probability
from policy.decision import Decision, DecisionPolicy
from policy.locked import build_locked_policy


@dataclass(frozen=True)
class RiskAssessment:
    """The scored result, in domain terms rather than wire terms.

    Carries the identity of the artifact that produced it, so a response can
    always state which model and feature contract it came from without the
    caller having to ask a second question.
    """

    transaction_id: str
    risk_probability: float
    decision: Decision
    explanation: TransactionExplanation
    history_transactions_used: int
    model_version: str
    feature_version: str


@dataclass(frozen=True)
class RiskScoringService:
    """Scores transactions against one frozen artifact.

    ``model`` is a view over ``artifact.pipeline`` rather than a second model:
    it exists so inference calls the same ``predict_fraud_probability`` the
    offline scripts call, against the artifact's own trees.
    """

    artifact: RiskForgeArtifact
    model: XGBoostFraudModel
    explainer: ModelExplainer
    policy: DecisionPolicy

    @classmethod
    def from_artifact(cls, artifact: RiskForgeArtifact) -> RiskScoringService:
        policy = build_locked_policy()

        _guard_policy_matches_artifact(artifact, policy)

        return cls(
            artifact=artifact,
            model=_model_view(artifact),
            explainer=build_model_explainer(artifact.pipeline),
            policy=policy,
        )

    def score(self, request: RiskScoreRequest) -> RiskAssessment:
        """Score one transaction against its caller-supplied prior history."""
        features = build_serving_features(request.transaction, request.history)

        raw_probability = predict_fraud_probability(self.model, features)

        calibrated = apply_calibrator(self.artifact.calibrator, raw_probability)

        # Full precision, deliberately: this exact float is what the policy
        # consumes. Rounding before decide() would let a probability such as
        # 0.0749999 cross the REVIEW threshold on paper while the model never
        # actually placed it there.
        probability = float(calibrated.iloc[0])

        explanation = explain_transaction(
            self.explainer,
            features,
            calibrated_risk_probability=probability,
            policy=self.policy,
            top_k=request.top_k,
        )

        return RiskAssessment(
            transaction_id=request.transaction.TransactionID,
            risk_probability=probability,
            decision=explanation.decision_explanation.decision,
            explanation=explanation,
            history_transactions_used=len(request.history),
            model_version=self.artifact.model_version,
            feature_version=self.artifact.feature_version,
        )


def to_response(assessment: RiskAssessment) -> RiskScoreResponse:
    """Render an assessment onto the wire contract, changing no value."""
    model = assessment.explanation.model_explanation
    decision = assessment.explanation.decision_explanation
    rule = decision_reason_code(assessment.decision.value)

    return RiskScoreResponse(
        transaction_id=assessment.transaction_id,
        risk_probability=assessment.risk_probability,
        decision=assessment.decision.value,
        reason_codes=[_reason_code_out(entry) for entry in model.risk_increasing],
        risk_decreasing_signals=[
            _reason_code_out(entry) for entry in model.risk_decreasing
        ],
        decision_reason_code=DecisionReasonCodeOut(
            code=rule.code,
            title=rule.title,
            type=rule.type,
            description=rule.description,
        ),
        decision_explanation=decision.explanation,
        model_detail=ModelDetailOut(
            base_value=model.base_value,
            raw_margin=model.raw_margin,
            raw_model_probability=model.raw_model_probability,
            attribution_space=model.attribution_space,
            attribution_space_note=ATTRIBUTION_SPACE_NOTE,
            total_risk_increasing=model.total_risk_increasing,
            total_risk_decreasing=model.total_risk_decreasing,
        ),
        policy_detail=PolicyDetailOut(
            review_threshold=decision.review_threshold,
            block_threshold=decision.block_threshold,
        ),
        model_version=assessment.model_version,
        feature_version=assessment.feature_version,
        policy_version=decision.policy_version,
        feature_count=len(MODEL_FEATURES),
        history_transactions_used=assessment.history_transactions_used,
    )


def _reason_code_out(entry: FeatureContribution) -> ReasonCodeOut:
    """The Phase 5 reason code for a contribution, with its measured value."""
    reason = reason_code_for_feature(entry.feature)

    return ReasonCodeOut(
        code=reason.code,
        title=reason.title,
        type=reason.type,
        description=reason.description,
        feature=entry.feature,
        contribution=entry.contribution,
        direction=entry.direction,
        n_transformed_columns=entry.n_transformed_columns,
    )


def _model_view(artifact: RiskForgeArtifact) -> XGBoostFraudModel:
    """Wrap the artifact's fitted pipeline in the type scoring expects.

    Nothing is fitted, copied or rebuilt: ``model`` is the artifact's own
    pipeline, so the trees that score a request are the trees locked offline.
    """
    return XGBoostFraudModel(
        model=artifact.pipeline,
        feature_columns=list(artifact.feature_columns),
        categorical_features=CATEGORICAL_FEATURES.copy(),
        numeric_features=NUMERIC_FEATURES.copy(),
        scale_pos_weight=artifact.scale_pos_weight,
    )


def _guard_policy_matches_artifact(
    artifact: RiskForgeArtifact,
    policy: DecisionPolicy,
) -> None:
    """Refuse to serve an artifact built under a different frozen policy.

    The artifact records the thresholds it was built with and the service
    enforces the ones in :mod:`policy.locked`. If those ever disagree, the
    decision and its own explanation would contradict each other, so this fails
    at startup instead of serving incoherent results.
    """
    config = policy.config

    mismatched: list[str] = []

    if artifact.policy_version != config.policy_version:
        mismatched.append(
            f"policy_version: artifact {artifact.policy_version!r}, "
            f"serving {config.policy_version!r}"
        )

    for name, artifact_value in (
        ("review_threshold", artifact.review_threshold),
        ("block_threshold", artifact.block_threshold),
    ):
        serving_value = float(getattr(config, name))

        if artifact_value != serving_value:
            mismatched.append(
                f"{name}: artifact {artifact_value}, serving {serving_value}"
            )

    if mismatched:
        raise ArtifactError(
            "The serving artifact was built under a different frozen policy "
            "than the one this service enforces ("
            + "; ".join(mismatched)
            + "). Rebuild the artifact rather than serving decisions whose "
            "thresholds disagree with their own explanation."
        )
