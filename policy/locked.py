"""The one definition of the frozen Phase 4 decision policy.

``policy-001`` and its two thresholds were selected in Phase 4 and are locked.
They were previously restated in each consuming script, which meant every new
consumer was a fresh opportunity to drift. This module is the single place the
frozen values are written down; scripts and the serving API build their policy
from here so a decision can always be traced to one definition.

Nothing is derived, optimised or validated at runtime beyond what
:class:`~policy.decision.PolicyConfig` already enforces.
"""

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
