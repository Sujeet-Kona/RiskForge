from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from explain.decision_explanation import DecisionExplanation, explain_decision
from explain.model_explanation import (
    DEFAULT_TOP_K,
    ModelExplanation,
    ModelExplainer,
)
from policy.decision import DecisionPolicy


TRANSACTION_ID_COLUMN = "TransactionID"


@dataclass(frozen=True)
class TransactionExplanation:
    """Both halves of an explanation for one transaction, kept separate.

    ``model_explanation`` answers "which features moved the model's fraud-risk
    score?". ``decision_explanation`` answers "why did that risk become ALLOW /
    REVIEW / BLOCK under the frozen policy?". They are reported side by side and
    deliberately never merged into one ranked list, because the strongest model
    signal on a transaction says nothing on its own about which band the policy
    put it in.
    """

    transaction_id: Any
    calibrated_risk_probability: float
    model_explanation: ModelExplanation
    decision_explanation: DecisionExplanation

    @property
    def decision(self) -> str:
        return self.decision_explanation.decision.value

    @property
    def top_reason_codes(self) -> tuple[str, ...]:
        """Model-signal codes first, then the single decision-rule code."""
        return (
            *self.model_explanation.top_reason_codes,
            self.decision_explanation.reason_code,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "calibrated_risk_probability": self.calibrated_risk_probability,
            "decision": self.decision,
            "decision_explanation": self.decision_explanation.to_dict(),
            "model_explanation": self.model_explanation.to_dict(),
            "top_reason_codes": list(self.top_reason_codes),
        }


def explain_transaction(
    explainer: ModelExplainer,
    transaction: pd.DataFrame,
    *,
    calibrated_risk_probability: float,
    policy: DecisionPolicy,
    top_k: int = DEFAULT_TOP_K,
) -> TransactionExplanation:
    """Explain one transaction end to end.

    ``calibrated_risk_probability`` is supplied by the caller from the locked
    Phase 3 pipeline (isotonic calibrator fitted on all of VALIDATION). This
    layer never refits, re-derives or approximates calibration, and it never
    looks at a label — the explanation is a function of the row and the frozen
    artifacts only.
    """
    if not isinstance(transaction, pd.DataFrame):
        raise TypeError("transaction must be a single-row pandas DataFrame.")

    model_explanation = explainer.explain(transaction, top_k=top_k)

    decision_explanation = explain_decision(policy, calibrated_risk_probability)

    transaction_id = (
        transaction[TRANSACTION_ID_COLUMN].iloc[0]
        if TRANSACTION_ID_COLUMN in transaction.columns
        else None
    )

    return TransactionExplanation(
        transaction_id=transaction_id,
        calibrated_risk_probability=float(calibrated_risk_probability),
        model_explanation=model_explanation,
        decision_explanation=decision_explanation,
    )
