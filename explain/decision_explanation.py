from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from explain.reason_codes import decision_reason_code
from policy.decision import Decision, DecisionPolicy


# Six decimals, not three: the frozen thresholds are 0.075 and 0.110, so a
# three-decimal rendering of a probability such as 0.0751 would print as
# "0.075" and appear to contradict the band the policy actually chose. The
# explanation has to be checkable against the decision at a glance.
PROBABILITY_FORMAT = ".6f"

MIN_PROBABILITY_DECIMALS = 6
MAX_PROBABILITY_DECIMALS = 12

ALLOW_TEMPLATE = (
    "Risk probability {probability} is below the REVIEW threshold {review}."
)
REVIEW_TEMPLATE = (
    "Risk probability {probability} falls between the REVIEW threshold "
    "{review} and BLOCK threshold {block}."
)
BLOCK_TEMPLATE = (
    "Risk probability {probability} meets or exceeds the BLOCK threshold "
    "{block}."
)

_TEMPLATES = {
    Decision.ALLOW: ALLOW_TEMPLATE,
    Decision.REVIEW: REVIEW_TEMPLATE,
    Decision.BLOCK: BLOCK_TEMPLATE,
}


@dataclass(frozen=True)
class DecisionExplanation:
    """The decision-level half of an explanation: why this band, not that one.

    This is a statement about the frozen policy's thresholds and nothing else.
    It carries no feature information, because which features moved the risk
    score is a separate question with a separate answer.
    """

    decision: Decision
    risk_probability: float
    review_threshold: float
    block_threshold: float
    policy_version: str
    reason_code: str
    reason_title: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "risk_probability": self.risk_probability,
            "review_threshold": self.review_threshold,
            "block_threshold": self.block_threshold,
            "policy_version": self.policy_version,
            "reason_code": self.reason_code,
            "reason_title": self.reason_title,
            "explanation": self.explanation,
        }


def explain_decision(
    policy: DecisionPolicy,
    risk_probability: float,
) -> DecisionExplanation:
    """Explain a calibrated risk probability under a frozen policy.

    The band comes from ``DecisionPolicy.decide`` and the thresholds come from
    the same ``PolicyConfig``, so the wording can never drift from the logic
    that actually ran. Nothing here re-implements the comparison.
    """
    if not isinstance(policy, DecisionPolicy):
        raise TypeError("policy must be a DecisionPolicy.")

    result = policy.decide(risk_probability)
    config = policy.config

    review = float(config.review_threshold)
    block = float(config.block_threshold)

    explanation = _TEMPLATES[result.decision].format(
        probability=_render_probability(result.risk_probability, (review, block)),
        review=format(review, PROBABILITY_FORMAT),
        block=format(block, PROBABILITY_FORMAT),
    )

    reason = decision_reason_code(result.decision.value)

    return DecisionExplanation(
        decision=result.decision,
        risk_probability=result.risk_probability,
        review_threshold=review,
        block_threshold=block,
        policy_version=result.policy_version,
        reason_code=reason.code,
        reason_title=reason.title,
        explanation=explanation,
    )


def _render_probability(probability: float, thresholds: tuple[float, ...]) -> str:
    """Render a probability so the sentence cannot appear to contradict itself.

    Six decimals is normally enough, but a probability such as 0.0749999 would
    print as "0.075000" and read as though it met the REVIEW threshold the
    policy just declined to apply. Digits are added only when the rounded text
    would land on a threshold the probability is not actually sitting on.
    """
    for decimals in range(MIN_PROBABILITY_DECIMALS, MAX_PROBABILITY_DECIMALS + 1):
        text = format(probability, f".{decimals}f")

        collided = [
            threshold
            for threshold in thresholds
            if text == format(threshold, f".{decimals}f")
            and probability != threshold
        ]

        if not collided:
            return text

    return repr(probability)
