from dataclasses import FrozenInstanceError, replace

import pytest

from explain.decision_explanation import (
    ALLOW_TEMPLATE,
    BLOCK_TEMPLATE,
    REVIEW_TEMPLATE,
    explain_decision,
)
from explain.reason_codes import DECISION_REASON_CODES
from policy.decision import Decision, DecisionPolicy, PolicyConfig


FROZEN_REVIEW_THRESHOLD = 0.075
FROZEN_BLOCK_THRESHOLD = 0.110
FROZEN_POLICY_VERSION = "policy-001"

EXPLANATION_FIELDS = {
    "decision",
    "risk_probability",
    "review_threshold",
    "block_threshold",
    "policy_version",
    "reason_code",
    "reason_title",
    "explanation",
}


def test_the_tiny_policy_carries_the_frozen_thresholds(tiny_policy: DecisionPolicy):
    """Everything below asserts against the real locked numbers, not copies."""
    assert tiny_policy.config.policy_version == FROZEN_POLICY_VERSION
    assert tiny_policy.config.review_threshold == FROZEN_REVIEW_THRESHOLD
    assert tiny_policy.config.block_threshold == FROZEN_BLOCK_THRESHOLD


# --- requirement 10: the wording matches each band exactly --------------------


@pytest.mark.parametrize(
    "probability,expected",
    [
        (
            0.042,
            "Risk probability 0.042000 is below the REVIEW threshold 0.075000.",
        ),
        (
            0.091,
            "Risk probability 0.091000 falls between the REVIEW threshold "
            "0.075000 and BLOCK threshold 0.110000.",
        ),
        (
            0.214,
            "Risk probability 0.214000 meets or exceeds the BLOCK threshold "
            "0.110000.",
        ),
    ],
    ids=["ALLOW", "REVIEW", "BLOCK"],
)
def test_each_band_produces_its_exact_sentence(
    tiny_policy: DecisionPolicy, probability: float, expected: str
):
    assert explain_decision(tiny_policy, probability).explanation == expected


def test_each_band_uses_its_own_template(tiny_policy: DecisionPolicy):
    allow = explain_decision(tiny_policy, 0.010)
    review = explain_decision(tiny_policy, 0.090)
    block = explain_decision(tiny_policy, 0.900)

    assert allow.explanation.startswith("Risk probability 0.010000 is below")
    assert review.explanation.startswith("Risk probability 0.090000 falls between")
    assert block.explanation.startswith("Risk probability 0.900000 meets or exceeds")
    assert len({ALLOW_TEMPLATE, REVIEW_TEMPLATE, BLOCK_TEMPLATE}) == 3


def test_the_band_always_agrees_with_the_policy_itself(tiny_policy: DecisionPolicy):
    for probability in (0.0, 0.04, 0.075, 0.09, 0.11, 0.5, 1.0):
        explained = explain_decision(tiny_policy, probability)

        assert explained.decision is tiny_policy.decide(probability).decision


@pytest.mark.parametrize(
    "probability,decision",
    [
        (0.0, Decision.ALLOW),
        (0.074999, Decision.ALLOW),
        (FROZEN_REVIEW_THRESHOLD, Decision.REVIEW),
        (0.109999, Decision.REVIEW),
        (FROZEN_BLOCK_THRESHOLD, Decision.BLOCK),
        (1.0, Decision.BLOCK),
    ],
    ids=[
        "zero",
        "just_below_review",
        "exactly_review",
        "just_below_block",
        "exactly_block",
        "one",
    ],
)
def test_the_band_boundaries_are_inclusive_at_the_lower_edge(
    tiny_policy: DecisionPolicy, probability: float, decision: Decision
):
    explained = explain_decision(tiny_policy, probability)

    assert explained.decision is decision
    assert explained.reason_code == DECISION_REASON_CODES[decision.value].code


def test_a_probability_on_a_threshold_is_not_rounded_off_it(
    tiny_policy: DecisionPolicy,
):
    """Exactly at T_REVIEW the sentence may name the threshold; below it may not."""
    at_threshold = explain_decision(tiny_policy, FROZEN_REVIEW_THRESHOLD)
    just_below = explain_decision(tiny_policy, 0.0749999)

    assert at_threshold.decision is Decision.REVIEW
    assert "0.075000 falls between" in at_threshold.explanation

    assert just_below.decision is Decision.ALLOW
    assert "0.0749999 is below" in just_below.explanation
    assert "Risk probability 0.075000 is below" not in just_below.explanation


# --- requirement 11: the actual frozen thresholds, never a copy ---------------


def test_the_explanation_carries_the_policy_thresholds(tiny_policy: DecisionPolicy):
    explained = explain_decision(tiny_policy, 0.090)

    assert explained.review_threshold == tiny_policy.config.review_threshold
    assert explained.block_threshold == tiny_policy.config.block_threshold
    assert explained.policy_version == tiny_policy.config.policy_version
    assert f"{FROZEN_REVIEW_THRESHOLD:.6f}" in explained.explanation
    assert f"{FROZEN_BLOCK_THRESHOLD:.6f}" in explained.explanation


def test_a_different_policy_rewrites_the_sentence(tiny_policy: DecisionPolicy):
    moved = replace(
        tiny_policy.config,
        policy_version="policy-test",
        review_threshold=0.20,
        block_threshold=0.40,
    )
    other = DecisionPolicy(moved)

    explained = explain_decision(other, 0.25)

    assert explained.decision is Decision.REVIEW
    assert explained.policy_version == "policy-test"
    assert explained.explanation == (
        "Risk probability 0.250000 falls between the REVIEW threshold "
        "0.200000 and BLOCK threshold 0.400000."
    )


def test_the_same_probability_can_change_band_under_a_different_policy(
    tiny_policy: DecisionPolicy,
):
    moved = replace(
        tiny_policy.config, review_threshold=0.02, block_threshold=0.05
    )
    other = DecisionPolicy(moved)

    assert explain_decision(tiny_policy, 0.030).decision is Decision.ALLOW
    assert explain_decision(other, 0.030).decision is Decision.REVIEW


def test_no_threshold_value_is_hardcoded_outside_the_policy_config():
    """The templates hold placeholders only; the numbers come from PolicyConfig."""
    for template in (ALLOW_TEMPLATE, REVIEW_TEMPLATE, BLOCK_TEMPLATE):
        assert "0.075" not in template
        assert "0.110" not in template
        assert "{probability}" in template


# --- separation from the model explanation ------------------------------------


def test_the_decision_explanation_carries_no_feature_information(
    tiny_policy: DecisionPolicy,
):
    payload = explain_decision(tiny_policy, 0.090).to_dict()

    assert set(payload) == EXPLANATION_FIELDS
    assert "feature" not in payload
    assert "features" not in payload
    assert "contribution" not in payload


def test_the_decision_reason_is_typed_as_a_rule_not_a_signal(
    tiny_policy: DecisionPolicy,
):
    explained = explain_decision(tiny_policy, 0.090)
    reason = DECISION_REASON_CODES[explained.decision.value]

    assert explained.reason_code == reason.code
    assert explained.reason_title == reason.title
    assert reason.type == "decision_rule"
    assert reason.features == ()


# --- determinism and validation -----------------------------------------------


def test_the_same_probability_produces_the_same_sentence(
    tiny_policy: DecisionPolicy,
):
    first = explain_decision(tiny_policy, 0.090).to_dict()
    second = explain_decision(tiny_policy, 0.090).to_dict()

    assert first == second


def test_a_non_policy_is_rejected():
    with pytest.raises(TypeError, match="must be a DecisionPolicy"):
        explain_decision(PolicyConfig, 0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("probability", [-0.1, 1.5, float("nan"), float("inf")])
def test_an_impossible_probability_is_rejected(
    tiny_policy: DecisionPolicy, probability: float
):
    with pytest.raises(ValueError):
        explain_decision(tiny_policy, probability)


@pytest.mark.parametrize("probability", ["0.5", None, True, [0.5]])
def test_a_non_numeric_probability_is_rejected(
    tiny_policy: DecisionPolicy, probability
):
    with pytest.raises(TypeError):
        explain_decision(tiny_policy, probability)


def test_the_explanation_is_frozen(tiny_policy: DecisionPolicy):
    explained = explain_decision(tiny_policy, 0.090)

    with pytest.raises(FrozenInstanceError):
        explained.explanation = "something else"  # type: ignore[misc]


def test_the_risk_probability_is_returned_unchanged(tiny_policy: DecisionPolicy):
    probability = 0.0913377

    assert (
        explain_decision(tiny_policy, probability).risk_probability == probability
    )
