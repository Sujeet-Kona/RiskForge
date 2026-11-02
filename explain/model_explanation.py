from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from explain.feature_attribution import (
    FeatureAttributionMap,
    build_feature_attribution_map,
    group_contributions,
)
from explain.reason_codes import reason_code_for_feature


DEFAULT_TOP_K = 3

PREPROCESSOR_STEP_NAME = "preprocessor"
CLASSIFIER_STEP_NAME = "classifier"

# Path-dependent TreeSHAP needs no background dataset, which is what makes the
# explanation a deterministic function of the fitted model and the input row.
FEATURE_PERTURBATION = "tree_path_dependent"
MODEL_OUTPUT = "raw"

RISK_INCREASING = "risk_increasing"
RISK_DECREASING = "risk_decreasing"
NEUTRAL = "neutral"

ATTRIBUTION_SPACE = "raw_margin_log_odds"

# XGBoost accumulates TreeSHAP values in float32, so the decomposition closes
# against the model's own margin to about 1e-5 rather than exactly.
FLOAT32_ADDITIVITY_TOLERANCE = 1e-4

ATTRIBUTION_SPACE_NOTE = (
    "Contributions are expressed in the XGBoost raw margin (log-odds) space, "
    "not in probability space. The base value plus the sum of every "
    "contribution equals the model's raw margin, and the raw model probability "
    "is sigmoid(margin). Because the locked isotonic calibration is monotonic, "
    "the sign of a contribution still tells you whether that feature pushed "
    "the deployed risk probability up or down; the magnitude does not carry "
    "over, since calibration rescales the margin non-linearly. Reason codes "
    "therefore describe model-level risk signals, not the calibration mapping."
)


@dataclass(frozen=True)
class FeatureContribution:
    """One contract feature's total share of the model's raw risk margin.

    ``contribution`` is the exact sum of the TreeSHAP values of every
    transformed column that encodes this feature — its imputed value, its
    missing indicator if it has one, and all of its one-hot levels.
    """

    feature: str
    contribution: float
    direction: str
    reason_code: str
    reason_title: str
    n_transformed_columns: int


@dataclass(frozen=True)
class ModelExplanation:
    """The model-level half of an explanation: what moved the risk score.

    This says nothing about the ALLOW / REVIEW / BLOCK action. That is the
    decision layer's job, and the two must not be merged: a feature can be the
    strongest risk driver on a transaction the policy still allows.
    """

    base_value: float
    raw_margin: float
    raw_model_probability: float
    attribution_space: str
    top_k: int
    contributions: tuple[FeatureContribution, ...]
    risk_increasing: tuple[FeatureContribution, ...]
    risk_decreasing: tuple[FeatureContribution, ...]
    total_risk_increasing: float
    total_risk_decreasing: float

    @property
    def top_reason_codes(self) -> tuple[str, ...]:
        """Reason codes for the features that pushed risk upward, best first."""
        return tuple(entry.reason_code for entry in self.risk_increasing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_value": self.base_value,
            "raw_margin": self.raw_margin,
            "raw_model_probability": self.raw_model_probability,
            "attribution_space": self.attribution_space,
            "top_k": self.top_k,
            "top_reason_codes": list(self.top_reason_codes),
            "risk_increasing": [_contribution_to_dict(c) for c in self.risk_increasing],
            "risk_decreasing": [_contribution_to_dict(c) for c in self.risk_decreasing],
            "total_risk_increasing": self.total_risk_increasing,
            "total_risk_decreasing": self.total_risk_decreasing,
        }


def _contribution_to_dict(entry: FeatureContribution) -> dict[str, Any]:
    return {
        "feature": entry.feature,
        "contribution": entry.contribution,
        "direction": entry.direction,
        "reason_code": entry.reason_code,
        "reason_title": entry.reason_title,
        "n_transformed_columns": entry.n_transformed_columns,
    }


@dataclass(frozen=True)
class ModelExplainer:
    """Explains the *actual fitted* RiskForge XGBoost model.

    Built once from the locked pipeline and reused. It holds no second model,
    no surrogate and no background dataset: the contributions come from
    TreeSHAP evaluated on the same booster that produces the production score.
    """

    pipeline: Pipeline
    attribution: FeatureAttributionMap
    explainer: Any

    @property
    def preprocessor(self) -> Any:
        return self.pipeline.named_steps[PREPROCESSOR_STEP_NAME]

    @property
    def classifier(self) -> Any:
        return self.pipeline.named_steps[CLASSIFIER_STEP_NAME]

    def explain(
        self,
        transaction: pd.DataFrame,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> ModelExplanation:
        """Explain a single transaction row.

        The row is passed through the *fitted* preprocessor, so missing numeric
        values receive the TRAIN median plus their missing indicator and missing
        categorical values receive the reserved ``__MISSING__`` level, exactly as
        they did at scoring time. No imputation logic is reimplemented here.
        """
        resolved_top_k = _validated_top_k(top_k)

        features = _single_row_features(transaction, self.attribution)

        transformed = self.preprocessor.transform(features)
        values = np.asarray(
            self.explainer.shap_values(transformed), dtype=float
        ).ravel()

        base_value = float(self.explainer.expected_value)
        raw_margin = float(values.sum() + base_value)

        raw_model_probability = float(
            self.pipeline.predict_proba(features)[0, 1]
        )

        grouped = group_contributions(self.attribution, values)

        entries = tuple(
            _build_entry(feature, contribution, self.attribution)
            for feature, contribution in grouped.items()
        )

        increasing = tuple(
            sorted(
                (entry for entry in entries if entry.contribution > 0.0),
                key=lambda entry: (-entry.contribution, entry.feature),
            )[:resolved_top_k]
        )
        decreasing = tuple(
            sorted(
                (entry for entry in entries if entry.contribution < 0.0),
                key=lambda entry: (entry.contribution, entry.feature),
            )[:resolved_top_k]
        )

        return ModelExplanation(
            base_value=base_value,
            raw_margin=raw_margin,
            raw_model_probability=raw_model_probability,
            attribution_space=ATTRIBUTION_SPACE,
            top_k=resolved_top_k,
            contributions=tuple(
                sorted(
                    entries,
                    key=lambda entry: (-abs(entry.contribution), entry.feature),
                )
            ),
            risk_increasing=increasing,
            risk_decreasing=decreasing,
            total_risk_increasing=float(
                sum(entry.contribution for entry in entries if entry.contribution > 0.0)
            ),
            total_risk_decreasing=float(
                sum(entry.contribution for entry in entries if entry.contribution < 0.0)
            ),
        )


def build_model_explainer(pipeline: Pipeline) -> ModelExplainer:
    """Attach a TreeSHAP explainer to the already-fitted locked pipeline.

    Nothing is fitted, retrained or resampled here. The explainer is read-only
    with respect to the model.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline must be a fitted sklearn Pipeline.")

    missing = [
        step
        for step in (PREPROCESSOR_STEP_NAME, CLASSIFIER_STEP_NAME)
        if step not in pipeline.named_steps
    ]

    if missing:
        raise ValueError(
            f"pipeline is missing required steps: {missing}. Expected the "
            "locked RiskForge preprocessing + classifier pipeline."
        )

    attribution = build_feature_attribution_map(
        pipeline.named_steps[PREPROCESSOR_STEP_NAME]
    )

    classifier = pipeline.named_steps[CLASSIFIER_STEP_NAME]

    if not hasattr(classifier, "predict_proba"):
        raise TypeError(
            "The pipeline's classifier cannot produce probabilities, so it is "
            "not the fitted RiskForge fraud model."
        )

    explainer = shap.TreeExplainer(
        classifier,
        feature_perturbation=FEATURE_PERTURBATION,
        model_output=MODEL_OUTPUT,
    )

    return ModelExplainer(
        pipeline=pipeline,
        attribution=attribution,
        explainer=explainer,
    )


def _build_entry(
    feature: str,
    contribution: float,
    attribution: FeatureAttributionMap,
) -> FeatureContribution:
    reason = reason_code_for_feature(feature)

    if contribution > 0.0:
        direction = RISK_INCREASING
    elif contribution < 0.0:
        direction = RISK_DECREASING
    else:
        direction = NEUTRAL

    return FeatureContribution(
        feature=feature,
        contribution=float(contribution),
        direction=direction,
        reason_code=reason.code,
        reason_title=reason.title,
        n_transformed_columns=len(attribution.columns_by_feature[feature]),
    )


def _single_row_features(
    transaction: pd.DataFrame,
    attribution: FeatureAttributionMap,
) -> pd.DataFrame:
    if not isinstance(transaction, pd.DataFrame):
        raise TypeError("transaction must be a single-row pandas DataFrame.")

    if len(transaction) != 1:
        raise ValueError(
            f"explain() takes exactly one transaction row; got {len(transaction)}."
        )

    missing = [
        feature
        for feature in attribution.contract_features
        if feature not in transaction.columns
    ]

    if missing:
        raise ValueError(f"Transaction row is missing features: {missing}")

    return transaction[attribution.contract_features]


def _validated_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    return top_k
