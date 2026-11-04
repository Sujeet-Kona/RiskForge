import inspect

import numpy as np
import pandas as pd
import pytest

from policy.costs import CostAssumptions
from policy.decision import Decision, DecisionPolicy, PolicyConfig
from policy.evaluation import (
    DEFAULT_POLICY_VERSION,
    candidate_thresholds,
    degenerate_baseline_costs,
    evaluate_policy,
    policy_outcome_to_dict,
    search_thresholds,
)


COSTS = CostAssumptions(
    false_negative_cost=100.0,
    false_positive_cost=10.0,
    review_cost=5.0,
    review_fraud_capture_rate=0.70,
)


def tiny_partition() -> tuple[pd.Series, pd.Series]:
    """Five rows whose cost can be worked out by hand."""
    probability = pd.Series([0.1, 0.3, 0.5, 0.8, 0.9], name="p")
    y_true = pd.Series([1, 0, 1, 0, 1], name="isFraud")

    return probability, y_true


def tiny_config(**overrides) -> PolicyConfig:
    values = {
        "policy_version": "policy-001",
        "review_threshold": 0.2,
        "block_threshold": 0.7,
        "costs": COSTS,
    }
    values.update(overrides)

    return PolicyConfig(**values)


def brute_force_cost(
    config: PolicyConfig,
    y_true: pd.Series,
    probability: pd.Series,
) -> float:
    """A row-by-row restatement of the cost model, from first principles."""
    costs = config.costs

    total = 0.0

    for label, score in zip(y_true.tolist(), probability.tolist()):
        if score < config.review_threshold:
            total += costs.false_negative_cost if label == 1 else 0.0
        elif score < config.block_threshold:
            total += costs.review_cost
            if label == 1:
                total += (
                    (1.0 - costs.review_fraud_capture_rate)
                    * costs.false_negative_cost
                )
        else:
            total += 0.0 if label == 1 else costs.false_positive_cost

    return total


def make_realistic_partition(
    n: int = 2_000,
    fraud_rate: float = 0.035,
    seed: int = 11,
) -> tuple[pd.Series, pd.Series]:
    """A concentrated, inflated score distribution like the real calibrated one."""
    rng = np.random.default_rng(seed)

    labels = pd.Series(
        (rng.random(n) < fraud_rate).astype(int),
        name="isFraud",
    )

    signal = labels.to_numpy() * 2.2 + rng.normal(0.0, 1.0, n)
    scores = np.clip(1.0 / (1.0 + np.exp(-(signal - 2.6))), 0.0, 1.0)

    return pd.Series(scores, name="p"), labels


# --- the cost model, verified by hand -------------------------------------


def test_evaluate_policy_matches_a_hand_computed_cost():
    probability, y_true = tiny_partition()

    outcome = evaluate_policy(tiny_config(), y_true, probability)

    assert outcome.allow_count == 1
    assert outcome.review_count == 2
    assert outcome.block_count == 2

    assert outcome.fraud_allowed == 1
    assert outcome.fraud_reviewed == 1
    assert outcome.fraud_blocked == 1
    assert outcome.legitimate_allowed == 0
    assert outcome.legitimate_reviewed == 1
    assert outcome.legitimate_blocked == 1

    # 1*100 + 2*5 + (1*0.3)*100 + 1*10
    assert outcome.total_expected_cost == pytest.approx(150.0)
    assert outcome.expected_cost_per_transaction == pytest.approx(30.0)


def test_rates_and_ratios_match_their_definitions():
    probability, y_true = tiny_partition()

    outcome = evaluate_policy(tiny_config(), y_true, probability)

    assert outcome.allow_rate == pytest.approx(1 / 5)
    assert outcome.review_rate == pytest.approx(2 / 5)
    assert outcome.block_rate == pytest.approx(2 / 5)

    assert outcome.precision == pytest.approx(1 / 2)
    assert outcome.review_fraud_rate == pytest.approx(1 / 2)
    assert outcome.false_positive_rate == pytest.approx(1 / 2)
    assert outcome.false_negative_rate == pytest.approx(1 / 3)
    assert outcome.fraud_missed_rate == pytest.approx(1 / 3)

    assert outcome.false_negatives == outcome.fraud_allowed
    assert outcome.false_positives == outcome.legitimate_blocked


def test_fraud_capture_rate_counts_only_the_assumed_review_share():
    probability, y_true = tiny_partition()

    outcome = evaluate_policy(tiny_config(), y_true, probability)

    assert outcome.expected_fraud_caught_in_review == pytest.approx(0.7)
    assert outcome.expected_fraud_escaped_in_review == pytest.approx(0.3)
    assert outcome.fraud_capture_rate == pytest.approx((1 + 0.7) / 3)


def test_review_effectiveness_of_one_captures_all_reviewed_fraud():
    probability, y_true = tiny_partition()

    costs = CostAssumptions(
        false_negative_cost=100.0,
        false_positive_cost=10.0,
        review_cost=5.0,
        review_fraud_capture_rate=1.0,
    )

    outcome = evaluate_policy(tiny_config(costs=costs), y_true, probability)

    assert outcome.expected_fraud_escaped_in_review == pytest.approx(0.0)
    assert outcome.fraud_capture_rate == pytest.approx(2 / 3)
    # 1*100 + 2*5 + 0 + 1*10
    assert outcome.total_expected_cost == pytest.approx(120.0)


def test_review_effectiveness_of_zero_charges_the_full_c_fn():
    probability, y_true = tiny_partition()

    costs = CostAssumptions(
        false_negative_cost=100.0,
        false_positive_cost=10.0,
        review_cost=5.0,
        review_fraud_capture_rate=0.0,
    )

    outcome = evaluate_policy(tiny_config(costs=costs), y_true, probability)

    assert outcome.expected_fraud_caught_in_review == pytest.approx(0.0)
    # 1*100 + 2*5 + 1*100 + 1*10
    assert outcome.total_expected_cost == pytest.approx(220.0)


def test_counts_always_partition_the_data_exactly():
    probability, y_true = make_realistic_partition()

    outcome = evaluate_policy(tiny_config(), y_true, probability)

    assert outcome.allow_count + outcome.review_count + outcome.block_count == (
        outcome.n_samples
    )
    assert (
        outcome.fraud_allowed
        + outcome.fraud_reviewed
        + outcome.fraud_blocked
    ) == outcome.fraud_total
    assert (
        outcome.legitimate_allowed
        + outcome.legitimate_reviewed
        + outcome.legitimate_blocked
    ) == outcome.legitimate_total
    assert outcome.fraud_total + outcome.legitimate_total == outcome.n_samples


@pytest.mark.parametrize(
    "review_threshold,block_threshold",
    [(0.0, 1.0), (0.25, 0.26), (0.5, 0.9), (0.001, 0.002)],
)
def test_evaluate_policy_agrees_with_a_row_by_row_cost(
    review_threshold, block_threshold
):
    probability, y_true = make_realistic_partition(n=600, seed=3)

    config = tiny_config(
        review_threshold=review_threshold,
        block_threshold=block_threshold,
    )

    outcome = evaluate_policy(config, y_true, probability)

    assert outcome.total_expected_cost == pytest.approx(
        brute_force_cost(config, y_true, probability),
        rel=1e-12,
    )


# --- degenerate baselines -------------------------------------------------


def test_degenerate_baselines_are_the_three_single_action_costs():
    _, y_true = tiny_partition()

    baselines = degenerate_baseline_costs(y_true, COSTS)

    assert baselines["always_allow"] == pytest.approx(3 * 100.0)
    assert baselines["always_review"] == pytest.approx(5 * 5.0 + 3 * 0.3 * 100.0)
    assert baselines["always_block"] == pytest.approx(2 * 10.0)


def test_degenerate_baselines_reject_a_non_binary_target():
    with pytest.raises(ValueError, match="only 0/1"):
        degenerate_baseline_costs(pd.Series([0, 2]), COSTS)


# --- the candidate grid ---------------------------------------------------


def test_candidate_thresholds_are_sorted_unique_and_in_range():
    probability, _ = make_realistic_partition()

    candidates = candidate_thresholds(probability)

    assert np.all(np.diff(candidates) > 0)
    assert candidates.min() >= 0.0
    assert candidates.max() <= 1.0
    assert candidates[0] == pytest.approx(0.0)
    assert candidates[-1] == pytest.approx(1.0)


def test_candidate_grid_is_concentrated_where_the_probability_mass_is():
    """94.6% of real calibrated VALIDATION scores fall below 0.10."""
    rng = np.random.default_rng(5)

    concentrated = pd.Series(
        np.concatenate([rng.uniform(0.0, 0.1, 9_460), rng.uniform(0.1, 1.0, 540)])
    )

    candidates = candidate_thresholds(concentrated)

    below = (candidates < 0.10).sum()

    assert below >= 20
    assert below / len(candidates) > 0.35


def test_candidate_thresholds_use_no_labels():
    """The grid comes from the score distribution, so it cannot leak."""
    probability, _ = make_realistic_partition(seed=17)

    assert np.array_equal(
        candidate_thresholds(probability),
        candidate_thresholds(probability),
    )


def test_candidate_thresholds_reject_out_of_range_scores():
    with pytest.raises(ValueError, match="between 0 and 1"):
        candidate_thresholds(pd.Series([0.2, 1.4]))


# --- the search -----------------------------------------------------------


def test_search_returns_the_true_minimum_over_the_whole_grid():
    probability, y_true = make_realistic_partition(n=400, seed=23)

    candidates = np.array(
        [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]
    )

    result = search_thresholds(
        probability,
        y_true,
        COSTS,
        candidates=candidates,
    )

    brute_force_best = min(
        brute_force_cost(
            tiny_config(review_threshold=low, block_threshold=high),
            y_true,
            probability,
        )
        for low in candidates
        for high in candidates
        if low < high
    )

    assert result.outcome.total_expected_cost == pytest.approx(
        brute_force_best,
        rel=1e-12,
    )


def test_search_fast_path_agrees_with_the_authoritative_outcome():
    probability, y_true = make_realistic_partition(n=800, seed=29)

    result = search_thresholds(probability, y_true, COSTS)

    assert result.winner_fast_path_cost == pytest.approx(
        result.outcome.total_expected_cost,
        rel=1e-12,
    )


def test_searched_policy_respects_the_threshold_invariant():
    probability, y_true = make_realistic_partition()

    result = search_thresholds(probability, y_true, COSTS)

    assert result.config.review_threshold < result.config.block_threshold
    assert 0.0 <= result.config.review_threshold <= 1.0
    assert 0.0 <= result.config.block_threshold <= 1.0


def test_search_stamps_the_requested_policy_version():
    probability, y_true = make_realistic_partition(n=300)

    result = search_thresholds(
        probability,
        y_true,
        COSTS,
        policy_version="policy-007",
    )

    assert result.config.policy_version == "policy-007"
    assert result.outcome.policy_version == "policy-007"


def test_search_defaults_to_policy_001():
    probability, y_true = make_realistic_partition(n=300)

    result = search_thresholds(probability, y_true, COSTS)

    assert result.config.policy_version == DEFAULT_POLICY_VERSION == "policy-001"


def test_search_carries_the_cost_configuration_into_the_frozen_policy():
    probability, y_true = make_realistic_partition(n=300)

    result = search_thresholds(probability, y_true, COSTS)

    assert result.config.costs is COSTS


def test_search_is_deterministic():
    probability, y_true = make_realistic_partition()

    first = search_thresholds(probability, y_true, COSTS)
    second = search_thresholds(probability, y_true, COSTS)

    assert first.config == second.config
    assert first.outcome == second.outcome


def test_search_is_invariant_to_row_order():
    """Sorting happens internally, so shuffling rows cannot change the winner."""
    probability, y_true = make_realistic_partition(n=600, seed=31)

    shuffled = np.random.default_rng(2).permutation(len(probability))

    original = search_thresholds(probability, y_true, COSTS)
    reordered = search_thresholds(
        probability.iloc[shuffled].reset_index(drop=True),
        y_true.iloc[shuffled].reset_index(drop=True),
        COSTS,
    )

    assert original.outcome.total_expected_cost == pytest.approx(
        reordered.outcome.total_expected_cost,
        rel=1e-12,
    )


def test_search_reports_how_much_of_the_space_it_covered():
    probability, y_true = make_realistic_partition(n=400)

    result = search_thresholds(probability, y_true, COSTS)

    assert result.candidate_threshold_values > 10
    assert result.feasible_pairs_evaluated > 0
    assert result.max_review_rate is None


def test_runners_up_are_cost_ordered_and_exclude_the_winner():
    probability, y_true = make_realistic_partition(n=500, seed=37)

    result = search_thresholds(probability, y_true, COSTS, runner_up_count=5)

    assert len(result.runners_up) <= 5

    costs = [entry.total_expected_cost for entry in result.runners_up]

    assert costs == sorted(costs)
    assert all(cost >= result.outcome.total_expected_cost for cost in costs)


def test_runners_up_can_be_disabled():
    probability, y_true = make_realistic_partition(n=300)

    result = search_thresholds(probability, y_true, COSTS, runner_up_count=0)

    assert result.runners_up == ()


# --- the review-rate constraint -------------------------------------------


def test_the_review_cap_is_off_by_default():
    probability, y_true = make_realistic_partition()

    unconstrained = search_thresholds(probability, y_true, COSTS)

    assert unconstrained.max_review_rate is None


def test_a_review_cap_is_obeyed_and_costs_more_than_the_unconstrained_optimum():
    probability, y_true = make_realistic_partition(n=3_000, seed=41)

    unconstrained = search_thresholds(probability, y_true, COSTS)
    capped = search_thresholds(
        probability,
        y_true,
        COSTS,
        max_review_rate=0.05,
    )

    assert capped.outcome.review_rate <= 0.05 + 1e-12
    assert capped.max_review_rate == 0.05
    assert capped.outcome.total_expected_cost >= (
        unconstrained.outcome.total_expected_cost - 1e-9
    )


def test_a_zero_review_cap_forces_a_two_way_policy():
    probability, y_true = make_realistic_partition(n=1_000, seed=59)

    capped = search_thresholds(probability, y_true, COSTS, max_review_rate=0.0)

    assert capped.outcome.review_count == 0
    assert capped.outcome.allow_count + capped.outcome.block_count == (
        capped.outcome.n_samples
    )


def test_an_out_of_range_review_cap_is_rejected():
    probability, y_true = make_realistic_partition(n=200)

    with pytest.raises(ValueError, match="between 0 and 1"):
        search_thresholds(probability, y_true, COSTS, max_review_rate=1.5)


# --- sensitivity to review effectiveness ----------------------------------


@pytest.mark.parametrize("effectiveness", [0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_the_search_runs_at_every_sensitivity_level(effectiveness):
    probability, y_true = make_realistic_partition(n=1_000, seed=43)

    costs = CostAssumptions(
        false_negative_cost=100.0,
        false_positive_cost=10.0,
        review_cost=5.0,
        review_fraud_capture_rate=effectiveness,
    )

    result = search_thresholds(probability, y_true, costs)

    assert result.config.review_threshold < result.config.block_threshold
    assert result.winner_fast_path_cost == pytest.approx(
        result.outcome.total_expected_cost,
        rel=1e-12,
    )


def test_a_higher_review_effectiveness_lowers_total_cost():
    """Same decisions, same data: only ``e`` changes, so cost must fall."""
    _, y_true = make_realistic_partition(n=1_000, seed=47)

    perfect = CostAssumptions(
        false_negative_cost=100.0,
        false_positive_cost=10.0,
        review_cost=5.0,
        review_fraud_capture_rate=1.0,
    )

    zero = CostAssumptions(
        false_negative_cost=100.0,
        false_positive_cost=10.0,
        review_cost=5.0,
        review_fraud_capture_rate=0.0,
    )

    probability = pd.Series(np.linspace(0.0, 1.0, len(y_true)))

    perfect_outcome = evaluate_policy(
        tiny_config(costs=perfect, review_threshold=0.0, block_threshold=1.0),
        y_true,
        probability,
    )
    zero_outcome = evaluate_policy(
        tiny_config(costs=zero, review_threshold=0.0, block_threshold=1.0),
        y_true,
        probability,
    )

    assert perfect_outcome.total_expected_cost < (
        zero_outcome.total_expected_cost
    )


# --- validation -----------------------------------------------------------


def test_search_rejects_mismatched_lengths():
    probability, y_true = make_realistic_partition(n=100)

    with pytest.raises(ValueError, match="same length"):
        search_thresholds(probability, y_true.iloc[:90], COSTS)


def test_search_rejects_out_of_range_probabilities():
    _, y_true = make_realistic_partition(n=50)

    with pytest.raises(ValueError, match="between 0 and 1"):
        search_thresholds(pd.Series(np.linspace(0, 1.2, 50)), y_true, COSTS)


def test_search_rejects_a_non_binary_target():
    probability, _ = make_realistic_partition(n=50)

    with pytest.raises(ValueError, match="only 0/1"):
        search_thresholds(probability, pd.Series(np.full(50, 3)), COSTS)


def test_search_rejects_a_single_candidate_threshold():
    probability, y_true = make_realistic_partition(n=50)

    with pytest.raises(ValueError, match="At least two candidate"):
        search_thresholds(probability, y_true, COSTS, candidates=np.array([0.5]))


# --- serialization --------------------------------------------------------


def test_policy_outcome_to_dict_covers_every_reported_field():
    probability, y_true = tiny_partition()

    values = policy_outcome_to_dict(evaluate_policy(tiny_config(), y_true, probability))

    required = {
        "policy_version",
        "review_threshold",
        "block_threshold",
        "allow_count",
        "review_count",
        "block_count",
        "allow_rate",
        "review_rate",
        "block_rate",
        "false_negatives",
        "false_positives",
        "fraud_allowed",
        "fraud_blocked",
        "legitimate_allowed",
        "legitimate_reviewed",
        "legitimate_blocked",
        "fraud_capture_rate",
        "precision",
        "false_positive_rate",
        "false_negative_rate",
        "total_expected_cost",
    }

    assert required <= set(values)
    assert all(
        isinstance(value, (float, int, str)) for value in values.values()
    )


def test_the_search_never_accepts_a_second_partition():
    """Thresholds come from one labelled set; there is no slot for another."""
    parameters = list(inspect.signature(search_thresholds).parameters)

    assert parameters.count("y_true") == 1
    assert parameters.count("probability") == 1
    assert not any("test" in name for name in parameters)


def test_the_frozen_policy_is_independent_of_the_object_that_produced_it():
    """A searched config behaves identically when rebuilt from its parts."""
    probability, y_true = make_realistic_partition(n=500, seed=53)

    result = search_thresholds(probability, y_true, COSTS)
    config = result.config

    rebuilt = PolicyConfig(
        policy_version=config.policy_version,
        review_threshold=config.review_threshold,
        block_threshold=config.block_threshold,
        costs=config.costs,
    )

    assert evaluate_policy(rebuilt, y_true, probability) == result.outcome

    original = DecisionPolicy(config).decide_many(probability).tolist()
    rebuilt_decisions = DecisionPolicy(rebuilt).decide_many(probability).tolist()

    assert original == rebuilt_decisions
    assert set(original) <= set(Decision)
