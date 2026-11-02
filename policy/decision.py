from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from policy.costs import CostAssumptions


DECISION_COLUMN_NAME = "decision"


class Decision(Enum):
    """The three business actions a transaction can receive."""

    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class PolicyConfig:
    """A complete, versionable description of one decision policy.

    Holds the version, both thresholds and the cost assumptions the thresholds
    were derived from, so a decision can always be traced back to the policy
    that produced it. The version lives only here — nothing else in the
    codebase hardcodes it.
    """

    policy_version: str
    review_threshold: float
    block_threshold: float
    costs: CostAssumptions

    def __post_init__(self) -> None:
        if not isinstance(self.policy_version, str):
            raise TypeError("policy_version must be a string.")

        if not self.policy_version.strip():
            raise ValueError("policy_version must not be blank.")

        for name in ("review_threshold", "block_threshold"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a real number.")

            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if float(self.review_threshold) >= float(self.block_threshold):
            raise ValueError(
                "review_threshold must be strictly less than block_threshold."
            )

        if not isinstance(self.costs, CostAssumptions):
            raise TypeError("costs must be a CostAssumptions instance.")


@dataclass(frozen=True)
class DecisionResult:
    """One decision, carrying everything needed to audit it later."""

    decision: Decision
    risk_probability: float
    policy_version: str


@dataclass(frozen=True)
class DecisionPolicy:
    """Turns a calibrated fraud probability into a business decision.

    This layer is deliberately independent of the model, the features and the
    calibrator: it consumes a bare probability and nothing else. Decisions are
    a pure function of ``(risk_probability, policy_version)`` — no randomness,
    no retraining, no database, no external service, no I/O.

    Band semantics are inclusive at the lower bound of each upper band, which
    matches the ``>= threshold`` convention already used by
    ``ml.models.evaluation.evaluate_predictions``::

        p <  T_REVIEW            -> ALLOW
        T_REVIEW <= p < T_BLOCK  -> REVIEW
        p >= T_BLOCK             -> BLOCK
    """

    config: PolicyConfig

    def decide(self, risk_probability: numbers.Real) -> DecisionResult:
        """Decide for a single calibrated probability."""
        probability = _validated_probability(risk_probability)

        return DecisionResult(
            decision=self._band(probability),
            risk_probability=probability,
            policy_version=self.config.policy_version,
        )

    def decide_many(self, risk_probability: pd.Series) -> pd.Series:
        """Decide for a whole partition, preserving row order and index."""
        if not isinstance(risk_probability, pd.Series):
            raise TypeError("risk_probability must be a pandas Series.")

        values = risk_probability.to_numpy(dtype=float)

        if values.size and not np.isfinite(values).all():
            raise ValueError(
                "risk_probability contains missing or non-finite values."
            )

        if values.size and ((values < 0.0) | (values > 1.0)).any():
            raise ValueError("risk_probability must be between 0 and 1.")

        review_threshold = float(self.config.review_threshold)
        block_threshold = float(self.config.block_threshold)

        bands = np.where(
            values < review_threshold,
            Decision.ALLOW,
            np.where(values < block_threshold, Decision.REVIEW, Decision.BLOCK),
        )

        return pd.Series(
            bands,
            index=risk_probability.index,
            name=DECISION_COLUMN_NAME,
        )

    def _band(self, probability: float) -> Decision:
        if probability < float(self.config.review_threshold):
            return Decision.ALLOW

        if probability < float(self.config.block_threshold):
            return Decision.REVIEW

        return Decision.BLOCK


def _validated_probability(risk_probability: numbers.Real) -> float:
    if isinstance(risk_probability, bool) or not isinstance(
        risk_probability, numbers.Real
    ):
        raise TypeError("risk_probability must be a real number.")

    probability = float(risk_probability)

    if not math.isfinite(probability):
        raise ValueError("risk_probability must be finite.")

    if not 0.0 <= probability <= 1.0:
        raise ValueError("risk_probability must be between 0 and 1.")

    return probability
