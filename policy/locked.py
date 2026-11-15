"""Frozen policy constants; their original selection rationale is not recorded."""

from __future__ import annotations

from policy.costs import EXPERIMENTAL_COST_ASSUMPTIONS
from policy.decision import DecisionPolicy, PolicyConfig


LOCKED_POLICY_VERSION = "policy-001"

# ALLOW: p < 0.075 | REVIEW: 0.075 <= p < 0.110 | BLOCK: p >= 0.110
LOCKED_REVIEW_THRESHOLD = 0.075
LOCKED_BLOCK_THRESHOLD = 0.110


def build_locked_policy() -> DecisionPolicy:
    """Return the frozen ``policy-001`` decision policy."""
    return DecisionPolicy(
        PolicyConfig(
            policy_version=LOCKED_POLICY_VERSION,
            review_threshold=LOCKED_REVIEW_THRESHOLD,
            block_threshold=LOCKED_BLOCK_THRESHOLD,
            costs=EXPERIMENTAL_COST_ASSUMPTIONS,
        )
    )
