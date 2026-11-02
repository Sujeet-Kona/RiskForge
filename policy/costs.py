from __future__ import annotations

import math
import numbers
from dataclasses import dataclass


ASSUMPTION_NOTICE = (
    "ASSUMPTION — synthetic project cost figures chosen for this experiment. "
    "They are NOT industry facts and make no claim about Visa, PayPal, banks, "
    "merchants, or any real payment network. The project holds no real "
    "merchant or payment-network cost data."
)


@dataclass(frozen=True)
class CostAssumptions:
    """The single home of every business-cost assumption in RiskForge.

    ``ASSUMPTION``: these are synthetic figures for this project experiment,
    not measured economics. See :data:`ASSUMPTION_NOTICE`.

    Symbol mapping used in the Phase 4 report:

    ==============================  ==========
    field                           symbol
    ==============================  ==========
    ``false_negative_cost``         ``C_FN``
    ``false_positive_cost``         ``C_FP``
    ``review_cost``                 ``C_REVIEW``
    ``review_fraud_capture_rate``   ``e``
    ==============================  ==========

    A false negative is a fraudulent transaction that is ALLOWED. A false
    positive is a legitimate transaction that is BLOCKED. A reviewed
    transaction always costs ``C_REVIEW``, and additionally costs
    ``C_FN`` with probability ``1 - e`` because manual review is assumed to
    intercept only a fraction ``e`` of the fraud routed to it.

    Costs are constant per transaction. They deliberately do not scale with
    ``TransactionAmt``: the policy interface receives only a calibrated
    probability, so the decision layer is amount-blind by design.
    """

    false_negative_cost: float
    false_positive_cost: float
    review_cost: float
    review_fraud_capture_rate: float

    def __post_init__(self) -> None:
        for name in (
            "false_negative_cost",
            "false_positive_cost",
            "review_cost",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a real number.")

            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

            if float(value) < 0.0:
                raise ValueError(f"{name} must not be negative.")

        rate = self.review_fraud_capture_rate

        if isinstance(rate, bool) or not isinstance(rate, numbers.Real):
            raise TypeError("review_fraud_capture_rate must be a real number.")

        if not math.isfinite(float(rate)):
            raise ValueError("review_fraud_capture_rate must be finite.")

        if not 0.0 <= float(rate) <= 1.0:
            raise ValueError(
                "review_fraud_capture_rate must be between 0 and 1."
            )

    @property
    def expected_review_escape_cost(self) -> float:
        """``C_FN`` charged on the share of reviewed fraud that gets away."""
        return self.false_negative_cost * (
            1.0 - float(self.review_fraud_capture_rate)
        )


EXPERIMENTAL_COST_ASSUMPTIONS = CostAssumptions(
    false_negative_cost=100.0,
    false_positive_cost=10.0,
    review_cost=5.0,
    review_fraud_capture_rate=0.70,
)
"""``ASSUMPTION``: the locked Phase 4 cost model.

For this project experiment we assume a missed fraud costs 100 units, a false
decline costs 10 units (a 10:1 ratio to missed fraud), and a manual review
costs 5 units — cheaper than a false decline, because a review can still save
the transaction. We further assume manual review intercepts 70% of the fraud
routed to it. None of these figures are industry facts.
"""
