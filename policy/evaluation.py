from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from policy.costs import CostAssumptions
from policy.decision import Decision, DecisionPolicy, PolicyConfig


DEFAULT_POLICY_VERSION = "policy-001"

# The fixed grid is dense below 0.10 because that is where the calibrated
# probability mass actually sits; a uniform grid over [0, 1] would waste almost
# every candidate on regions holding a handful of rows.
_FIXED_GRID = np.concatenate(
    [
        [0.0],
        np.arange(0.005, 0.1001, 0.005),
        np.arange(0.11, 0.3001, 0.01),
        [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0],
    ]
)


@dataclass(frozen=True)
class PolicyOutcome:
    """Every operational and cost consequence of one policy on one partition.

    ``false_negatives`` is a fraudulent transaction that was ALLOWED, and
    ``false_positives`` is a legitimate transaction that was BLOCKED — the two
    definitions fixed by the Phase 4 cost model. A legitimate transaction sent
    to REVIEW is *not* a false positive; it is reported separately as
    ``legitimate_reviewed`` and charged ``C_REVIEW``.

    ``fraud_capture_rate`` and the two ``expected_fraud_*_in_review`` fields are
    expectations under the ``ASSUMPTION`` that manual review intercepts the
    fraction ``review_fraud_capture_rate`` of the fraud routed to it. They are
    not measured quantities.
    """

    policy_version: str
    review_threshold: float
    block_threshold: float

    n_samples: int
    fraud_total: int
    legitimate_total: int

    allow_count: int
    review_count: int
    block_count: int
    allow_rate: float
    review_rate: float
    block_rate: float

    fraud_allowed: int
    fraud_reviewed: int
    fraud_blocked: int
    legitimate_allowed: int
    legitimate_reviewed: int
    legitimate_blocked: int

    expected_fraud_caught_in_review: float
    expected_fraud_escaped_in_review: float

    fraud_capture_rate: float
    fraud_missed_rate: float
    precision: float
    review_fraud_rate: float
    false_positive_rate: float
    false_negative_rate: float

    total_expected_cost: float
    expected_cost_per_transaction: float

    @property
    def false_negatives(self) -> int:
        """Fraud that was ALLOWED and therefore definitively missed."""
        return self.fraud_allowed

    @property
    def false_positives(self) -> int:
        """Legitimate transactions that were BLOCKED."""
        return self.legitimate_blocked


@dataclass(frozen=True)
class ThresholdSearchResult:
    """The winning policy plus enough context to audit how it won."""

    config: PolicyConfig
    outcome: PolicyOutcome
    candidate_threshold_values: int
    feasible_pairs_evaluated: int
    max_review_rate: float | None
    runners_up: tuple[PolicyOutcome, ...]
    winner_fast_path_cost: float
    """Cost from the O(1) prefix-sum path. Must equal ``outcome``'s cost."""


def _validate_partition(
    y_true: pd.Series,
    probability: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    if len(y_true) != len(probability):
        raise ValueError("y_true and probability must have the same length.")

    if len(y_true) == 0:
        raise ValueError("Policy evaluation requires at least one row.")

    if y_true.isna().any():
        raise ValueError("y_true contains missing values.")

    if not y_true.isin([0, 1]).all():
        raise ValueError("y_true must contain only 0/1 values.")

    if probability.isna().any():
        raise ValueError("probability contains missing values.")

    values = probability.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("probability contains non-finite values.")

    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("probability must be between 0 and 1.")

    return y_true.to_numpy(dtype=int), values


def evaluate_policy(
    config: PolicyConfig,
    y_true: pd.Series,
    probability: pd.Series,
) -> PolicyOutcome:
    """Score one frozen policy against one labelled partition.

    This is the readable, authoritative definition of the cost model. The
    threshold search uses an algebraically equivalent fast path; unit tests
    assert the two agree.
    """
    labels, values = _validate_partition(y_true, probability)

    decisions = DecisionPolicy(config).decide_many(probability).to_numpy()

    fraud = labels == 1
    legitimate = ~fraud

    allowed = decisions == Decision.ALLOW
    reviewed = decisions == Decision.REVIEW
    blocked = decisions == Decision.BLOCK

    n = len(labels)
    fraud_total = int(fraud.sum())
    legitimate_total = int(legitimate.sum())

    allow_count = int(allowed.sum())
    review_count = int(reviewed.sum())
    block_count = int(blocked.sum())

    fraud_allowed = int((fraud & allowed).sum())
    fraud_reviewed = int((fraud & reviewed).sum())
    fraud_blocked = int((fraud & blocked).sum())
    legitimate_allowed = int((legitimate & allowed).sum())
    legitimate_reviewed = int((legitimate & reviewed).sum())
    legitimate_blocked = int((legitimate & blocked).sum())

    return _assemble_outcome(
        config=config,
        n=n,
        fraud_total=fraud_total,
        legitimate_total=legitimate_total,
        allow_count=allow_count,
        review_count=review_count,
        block_count=block_count,
        fraud_allowed=fraud_allowed,
        fraud_reviewed=fraud_reviewed,
        fraud_blocked=fraud_blocked,
        legitimate_allowed=legitimate_allowed,
        legitimate_reviewed=legitimate_reviewed,
        legitimate_blocked=legitimate_blocked,
    )


def _assemble_outcome(
    *,
    config: PolicyConfig,
    n: int,
    fraud_total: int,
    legitimate_total: int,
    allow_count: int,
    review_count: int,
    block_count: int,
    fraud_allowed: int,
    fraud_reviewed: int,
    fraud_blocked: int,
    legitimate_allowed: int,
    legitimate_reviewed: int,
    legitimate_blocked: int,
) -> PolicyOutcome:
    costs = config.costs
    capture_rate = float(costs.review_fraud_capture_rate)

    expected_caught_in_review = fraud_reviewed * capture_rate
    expected_escaped_in_review = fraud_reviewed - expected_caught_in_review

    total_cost = (
        fraud_allowed * costs.false_negative_cost
        + review_count * costs.review_cost
        + expected_escaped_in_review * costs.false_negative_cost
        + legitimate_blocked * costs.false_positive_cost
    )

    fraud_caught = fraud_blocked + expected_caught_in_review

    return PolicyOutcome(
        policy_version=config.policy_version,
        review_threshold=float(config.review_threshold),
        block_threshold=float(config.block_threshold),
        n_samples=n,
        fraud_total=fraud_total,
        legitimate_total=legitimate_total,
        allow_count=allow_count,
        review_count=review_count,
        block_count=block_count,
        allow_rate=allow_count / n,
        review_rate=review_count / n,
        block_rate=block_count / n,
        fraud_allowed=fraud_allowed,
        fraud_reviewed=fraud_reviewed,
        fraud_blocked=fraud_blocked,
        legitimate_allowed=legitimate_allowed,
        legitimate_reviewed=legitimate_reviewed,
        legitimate_blocked=legitimate_blocked,
        expected_fraud_caught_in_review=expected_caught_in_review,
        expected_fraud_escaped_in_review=expected_escaped_in_review,
        fraud_capture_rate=(fraud_caught / fraud_total) if fraud_total else 0.0,
        fraud_missed_rate=(fraud_allowed / fraud_total) if fraud_total else 0.0,
        precision=(fraud_blocked / block_count) if block_count else 0.0,
        review_fraud_rate=(fraud_reviewed / review_count) if review_count else 0.0,
        false_positive_rate=(
            legitimate_blocked / legitimate_total if legitimate_total else 0.0
        ),
        false_negative_rate=(fraud_allowed / fraud_total) if fraud_total else 0.0,
        total_expected_cost=float(total_cost),
        expected_cost_per_transaction=float(total_cost) / n,
    )


def degenerate_baseline_costs(
    y_true: pd.Series,
    costs: CostAssumptions,
) -> dict[str, float]:
    """Total cost of the three single-action policies.

    These are the floors any three-way policy has to beat. They are not
    expressible as a ``PolicyConfig``, because ``review_threshold <
    block_threshold`` is required, so they are computed directly.
    """
    if y_true.isna().any():
        raise ValueError("y_true contains missing values.")

    if not y_true.isin([0, 1]).all():
        raise ValueError("y_true must contain only 0/1 values.")

    if len(y_true) == 0:
        raise ValueError("Baseline costs require at least one row.")

    n = len(y_true)
    fraud_total = int(y_true.sum())
    legitimate_total = n - fraud_total

    capture_rate = float(costs.review_fraud_capture_rate)

    return {
        "always_allow": fraud_total * costs.false_negative_cost,
        "always_review": (
            n * costs.review_cost
            + fraud_total * (1.0 - capture_rate) * costs.false_negative_cost
        ),
        "always_block": legitimate_total * costs.false_positive_cost,
    }


def candidate_thresholds(
    probability: pd.Series,
    *,
    quantile_step: float = 1.0,
) -> np.ndarray:
    """Build a search grid adapted to the actual score distribution.

    Combines the empirical percentiles of ``probability`` with a fixed
    interpretable grid. Uses the score *distribution* only — no labels — so
    constructing the grid cannot leak.
    """
    if not 0.0 < quantile_step < 100.0:
        raise ValueError("quantile_step must be between 0 and 100.")

    if probability.isna().any():
        raise ValueError("probability contains missing values.")

    values = probability.to_numpy(dtype=float)

    if values.size == 0:
        raise ValueError("candidate_thresholds requires at least one value.")

    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("probability must be between 0 and 1.")

    percentiles = np.arange(quantile_step, 100.0, quantile_step)
    quantile_candidates = np.percentile(values, percentiles)

    combined = np.concatenate([quantile_candidates, _FIXED_GRID])
    combined = np.clip(np.round(combined, 6), 0.0, 1.0)

    return np.unique(combined)


def search_thresholds(
    probability: pd.Series,
    y_true: pd.Series,
    costs: CostAssumptions,
    *,
    candidates: np.ndarray | None = None,
    max_review_rate: float | None = None,
    policy_version: str = DEFAULT_POLICY_VERSION,
    runner_up_count: int = 5,
) -> ThresholdSearchResult:
    """Find the minimum-expected-cost ``(T_REVIEW, T_BLOCK)`` pair.

    Sorting the rows once and using cumulative fraud/legitimate counts makes
    the cost of any band pair an O(1) lookup, so the whole quadratic grid is
    searched exactly rather than sampled.

    Candidates are collapsed by the number of rows they place below the
    threshold, because two thresholds in the same gap between distinct
    probability values produce identical decisions. Ties on cost are broken
    deterministically by fewest reviews, then lowest ``T_REVIEW``, then lowest
    ``T_BLOCK``.

    ``max_review_rate`` is an *optional operational* constraint, disabled by
    default. It is not part of the primary Phase 4 objective.
    """
    if runner_up_count < 0:
        raise ValueError("runner_up_count must not be negative.")

    if max_review_rate is not None and not 0.0 <= max_review_rate <= 1.0:
        raise ValueError("max_review_rate must be between 0 and 1.")

    labels, values = _validate_partition(y_true, probability)

    if candidates is None:
        candidates = candidate_thresholds(probability)

    candidates = np.unique(np.asarray(candidates, dtype=float))

    if candidates.size < 2:
        raise ValueError("At least two candidate thresholds are required.")

    if ((candidates < 0.0) | (candidates > 1.0)).any():
        raise ValueError("Candidate thresholds must be between 0 and 1.")

    n = len(values)

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_labels = labels[order]

    cumulative_fraud = np.concatenate(([0], np.cumsum(sorted_labels)))
    cumulative_legitimate = np.concatenate(
        ([0], np.cumsum(1 - sorted_labels))
    )

    total_fraud = int(cumulative_fraud[n])
    total_legitimate = int(cumulative_legitimate[n])

    cuts = np.searchsorted(sorted_values, candidates, side="left")

    lowest: dict[int, float] = {}
    highest: dict[int, float] = {}

    for threshold, cut in zip(candidates.tolist(), cuts.tolist()):
        if cut not in lowest:
            lowest[cut] = threshold
        highest[cut] = threshold

    unique_cuts = np.array(sorted(lowest), dtype=int)
    has_gap = np.array([lowest[cut] < highest[cut] for cut in unique_cuts])

    cut_review = unique_cuts[:, None]
    cut_block = unique_cuts[None, :]

    review_count = cut_block - cut_review
    fraud_allowed = cumulative_fraud[cut_review]
    fraud_reviewed = cumulative_fraud[cut_block] - cumulative_fraud[cut_review]
    legitimate_blocked = total_legitimate - cumulative_legitimate[cut_block]

    capture_rate = float(costs.review_fraud_capture_rate)

    cost = (
        fraud_allowed * costs.false_negative_cost
        + review_count * costs.review_cost
        + fraud_reviewed * (1.0 - capture_rate) * costs.false_negative_cost
        + legitimate_blocked * costs.false_positive_cost
    )

    # A pair is a real policy only when T_REVIEW < T_BLOCK. Equal cut indices
    # are admissible when the candidate grid spans a gap in the score
    # distribution, so both thresholds can sit inside that gap.
    feasible = (cut_review < cut_block) | (
        (cut_review == cut_block) & has_gap[:, None]
    )

    if max_review_rate is not None:
        feasible &= (review_count / n) <= max_review_rate

    cost = np.where(feasible, cost, np.inf)

    # Rank every feasible pair by cost, then break exact ties deterministically
    # by fewest reviews, lowest T_REVIEW, lowest T_BLOCK. Runners-up are the
    # next-best distinct policies, not other winners tied on cost.
    feasible_positions = np.flatnonzero(feasible.ravel())
    ranked_cost = cost.ravel()[feasible_positions]
    ranked_reviews = review_count.ravel()[feasible_positions]
    ranked_review_cut = np.broadcast_to(cut_review, cost.shape).ravel()[
        feasible_positions
    ]
    ranked_block_cut = np.broadcast_to(cut_block, cost.shape).ravel()[
        feasible_positions
    ]

    ordered = np.lexsort(
        (ranked_block_cut, ranked_review_cut, ranked_reviews, ranked_cost)
    )
    ranked_positions = feasible_positions[ordered]

    def thresholds_for(
        position: int,
    ) -> tuple[float, float]:
        row, column = np.unravel_index(int(position), cost.shape)

        review_cut = int(unique_cuts[row])
        block_cut = int(unique_cuts[column])

        return lowest[review_cut], (
            lowest[block_cut] if review_cut != block_cut else highest[block_cut]
        )

    review_threshold, block_threshold = thresholds_for(ranked_positions[0])

    config = PolicyConfig(
        policy_version=policy_version,
        review_threshold=review_threshold,
        block_threshold=block_threshold,
        costs=costs,
    )

    outcome = evaluate_policy(config, y_true, probability)

    runners_up = tuple(
        evaluate_policy(
            PolicyConfig(
                policy_version=policy_version,
                review_threshold=runner_review,
                block_threshold=runner_block,
                costs=costs,
            ),
            y_true,
            probability,
        )
        for runner_review, runner_block in (
            thresholds_for(position)
            for position in ranked_positions[1 : runner_up_count + 1]
        )
    )

    return ThresholdSearchResult(
        config=config,
        outcome=outcome,
        candidate_threshold_values=int(candidates.size),
        feasible_pairs_evaluated=int(feasible.sum()),
        max_review_rate=max_review_rate,
        runners_up=runners_up,
        winner_fast_path_cost=float(ranked_cost.min()),
    )


def policy_outcome_to_dict(outcome: PolicyOutcome) -> dict[str, float | int | str]:
    """Convert an outcome to a serializable dictionary."""
    return {
        "policy_version": outcome.policy_version,
        "review_threshold": outcome.review_threshold,
        "block_threshold": outcome.block_threshold,
        "n_samples": outcome.n_samples,
        "fraud_total": outcome.fraud_total,
        "legitimate_total": outcome.legitimate_total,
        "allow_count": outcome.allow_count,
        "review_count": outcome.review_count,
        "block_count": outcome.block_count,
        "allow_rate": outcome.allow_rate,
        "review_rate": outcome.review_rate,
        "block_rate": outcome.block_rate,
        "fraud_allowed": outcome.fraud_allowed,
        "fraud_reviewed": outcome.fraud_reviewed,
        "fraud_blocked": outcome.fraud_blocked,
        "legitimate_allowed": outcome.legitimate_allowed,
        "legitimate_reviewed": outcome.legitimate_reviewed,
        "legitimate_blocked": outcome.legitimate_blocked,
        "false_negatives": outcome.false_negatives,
        "false_positives": outcome.false_positives,
        "expected_fraud_caught_in_review": outcome.expected_fraud_caught_in_review,
        "expected_fraud_escaped_in_review": outcome.expected_fraud_escaped_in_review,
        "fraud_capture_rate": outcome.fraud_capture_rate,
        "fraud_missed_rate": outcome.fraud_missed_rate,
        "precision": outcome.precision,
        "review_fraud_rate": outcome.review_fraud_rate,
        "false_positive_rate": outcome.false_positive_rate,
        "false_negative_rate": outcome.false_negative_rate,
        "total_expected_cost": outcome.total_expected_cost,
        "expected_cost_per_transaction": outcome.expected_cost_per_transaction,
    }
