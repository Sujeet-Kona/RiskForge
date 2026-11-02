import ast
import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from policy.costs import (
    ASSUMPTION_NOTICE,
    EXPERIMENTAL_COST_ASSUMPTIONS,
    CostAssumptions,
)
from policy.decision import (
    DECISION_COLUMN_NAME,
    Decision,
    DecisionPolicy,
    DecisionResult,
    PolicyConfig,
)


REVIEW_THRESHOLD = 0.20
BLOCK_THRESHOLD = 0.70


def make_config(**overrides) -> PolicyConfig:
    values = {
        "policy_version": "policy-001",
        "review_threshold": REVIEW_THRESHOLD,
        "block_threshold": BLOCK_THRESHOLD,
        "costs": EXPERIMENTAL_COST_ASSUMPTIONS,
    }
    values.update(overrides)

    return PolicyConfig(**values)


def make_policy(**overrides) -> DecisionPolicy:
    return DecisionPolicy(make_config(**overrides))


# --- STEP 9.1 / 9.3 / 9.5 — the three bands -------------------------------


def test_probability_below_the_review_threshold_is_allowed():
    result = make_policy().decide(0.199999)

    assert result.decision is Decision.ALLOW


def test_probability_between_the_thresholds_is_reviewed():
    result = make_policy().decide(0.45)

    assert result.decision is Decision.REVIEW


def test_probability_above_the_block_threshold_is_blocked():
    result = make_policy().decide(0.700001)

    assert result.decision is Decision.BLOCK


# --- STEP 9.2 / 9.4 — inclusive lower bounds ------------------------------


def test_probability_exactly_at_the_review_threshold_is_reviewed():
    result = make_policy().decide(REVIEW_THRESHOLD)

    assert result.decision is Decision.REVIEW


def test_probability_exactly_at_the_block_threshold_is_blocked():
    result = make_policy().decide(BLOCK_THRESHOLD)

    assert result.decision is Decision.BLOCK


def test_band_boundaries_match_the_existing_evaluation_convention():
    """``evaluate_predictions`` uses ``>= threshold``; the policy must agree."""
    policy = make_policy()

    assert policy.decide(REVIEW_THRESHOLD).decision is Decision.REVIEW
    assert policy.decide(np.nextafter(REVIEW_THRESHOLD, 0.0)).decision is (
        Decision.ALLOW
    )
    assert policy.decide(BLOCK_THRESHOLD).decision is Decision.BLOCK
    assert policy.decide(np.nextafter(BLOCK_THRESHOLD, 0.0)).decision is (
        Decision.REVIEW
    )


# --- STEP 9.6 / 9.7 — closed domain ---------------------------------------


def test_probability_zero_is_allowed():
    result = make_policy().decide(0.0)

    assert result.decision is Decision.ALLOW
    assert result.risk_probability == 0.0


def test_probability_one_is_blocked():
    result = make_policy().decide(1.0)

    assert result.decision is Decision.BLOCK
    assert result.risk_probability == 1.0


def test_probability_one_is_blocked_even_when_the_block_threshold_is_one():
    policy = make_policy(review_threshold=0.5, block_threshold=1.0)

    assert policy.decide(1.0).decision is Decision.BLOCK
    assert policy.decide(0.999).decision is Decision.REVIEW


# --- STEP 9.8 / 9.9 — rejected probabilities ------------------------------


def test_negative_probability_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        make_policy().decide(-0.001)


def test_probability_above_one_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        make_policy().decide(1.001)


def test_nan_probability_is_rejected_not_silently_blocked():
    """NaN fails every ``<`` comparison, so it would fall through to BLOCK."""
    with pytest.raises(ValueError, match="finite"):
        make_policy().decide(float("nan"))


def test_infinite_probability_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        make_policy().decide(float("inf"))


def test_non_numeric_probability_is_rejected():
    with pytest.raises(TypeError, match="real number"):
        make_policy().decide("0.5")


def test_boolean_probability_is_rejected():
    with pytest.raises(TypeError, match="real number"):
        make_policy().decide(True)


# --- STEP 9.10 — threshold ordering ---------------------------------------


def test_review_threshold_equal_to_block_threshold_is_rejected():
    with pytest.raises(ValueError, match="strictly less than"):
        make_config(review_threshold=0.5, block_threshold=0.5)


def test_review_threshold_above_block_threshold_is_rejected():
    with pytest.raises(ValueError, match="strictly less than"):
        make_config(review_threshold=0.8, block_threshold=0.3)


def test_thresholds_outside_the_unit_interval_are_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        make_config(review_threshold=-0.1, block_threshold=0.5)

    with pytest.raises(ValueError, match="between 0 and 1"):
        make_config(review_threshold=0.1, block_threshold=1.5)


def test_degenerate_but_valid_threshold_orderings_are_accepted():
    """T_REVIEW=0 and T_BLOCK=1 are extreme, not invalid."""
    assert make_config(review_threshold=0.0, block_threshold=1.0)
    assert make_config(review_threshold=0.0, block_threshold=0.001)


# --- STEP 9.11 / 9.12 — configuration is preserved ------------------------


def test_policy_version_is_preserved_on_every_decision():
    policy = make_policy(policy_version="policy-042")

    for probability in (0.0, 0.3, 0.9):
        result = policy.decide(probability)

        assert isinstance(result, DecisionResult)
        assert result.policy_version == "policy-042"


def test_blank_policy_version_is_rejected():
    with pytest.raises(ValueError, match="must not be blank"):
        make_config(policy_version="   ")


def test_cost_configuration_is_preserved_verbatim():
    costs = CostAssumptions(
        false_negative_cost=100.0,
        false_positive_cost=10.0,
        review_cost=5.0,
        review_fraud_capture_rate=0.70,
    )

    config = make_config(costs=costs)

    assert config.costs is costs
    assert config.costs.false_negative_cost == 100.0
    assert config.costs.false_positive_cost == 10.0
    assert config.costs.review_cost == 5.0
    assert config.costs.review_fraud_capture_rate == 0.70


def test_config_rejects_a_non_cost_assumptions_object():
    with pytest.raises(TypeError, match="CostAssumptions"):
        make_config(costs={"false_negative_cost": 100.0})


def test_config_is_frozen_and_hashable():
    config = make_config()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.review_threshold = 0.9

    assert hash(config) == hash(make_config())


# --- cost assumption validation -------------------------------------------


def test_negative_costs_are_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        CostAssumptions(
            false_negative_cost=-1.0,
            false_positive_cost=10.0,
            review_cost=5.0,
            review_fraud_capture_rate=0.7,
        )


@pytest.mark.parametrize("rate", [-0.01, 1.01])
def test_review_capture_rate_outside_the_unit_interval_is_rejected(rate):
    with pytest.raises(ValueError, match="between 0 and 1"):
        CostAssumptions(
            false_negative_cost=100.0,
            false_positive_cost=10.0,
            review_cost=5.0,
            review_fraud_capture_rate=rate,
        )


def test_expected_review_escape_cost_is_the_uncaptured_share_of_c_fn():
    costs = CostAssumptions(
        false_negative_cost=100.0,
        false_positive_cost=10.0,
        review_cost=5.0,
        review_fraud_capture_rate=0.70,
    )

    assert costs.expected_review_escape_cost == pytest.approx(30.0)


def test_cost_assumptions_are_labeled_as_an_assumption_not_a_fact():
    """STEP 13: the code itself must not read as an industry claim."""
    notice = ASSUMPTION_NOTICE

    assert "ASSUMPTION" in notice
    assert "NOT industry facts" in notice

    for name in ("Visa", "PayPal", "banks", "merchants"):
        assert name in notice

    source = Path(inspect.getfile(CostAssumptions)).read_text()

    assert "ASSUMPTION" in source
    assert "Industry fraud costs are" not in source


# --- STEP 10 — determinism ------------------------------------------------


def test_the_same_probability_and_version_always_give_the_same_decision():
    policy = make_policy()

    first = policy.decide(0.4242)
    second = policy.decide(0.4242)

    assert first == second
    assert first.decision is second.decision
    assert first.risk_probability == second.risk_probability
    assert first.policy_version == second.policy_version


def test_two_independently_built_policies_agree():
    probabilities = np.linspace(0.0, 1.0, 201)

    left = [make_policy().decide(p).decision for p in probabilities]
    right = [make_policy().decide(p).decision for p in probabilities]

    assert left == right


def test_decide_does_not_depend_on_any_call_order_or_prior_state():
    policy = make_policy()

    before = policy.decide(0.5).decision

    for probability in (0.0, 1.0, 0.99, 0.01, 0.2, 0.7):
        policy.decide(probability)

    assert policy.decide(0.5).decision is before


def test_the_policy_layer_performs_no_io_and_holds_no_randomness():
    """STEP 10: no randomness, no DB, no network, no file access.

    This cannot be a substring scan, because the module's own docstrings describe
    the absence of exactly these things ("no randomness, no database") and a naive
    match would trip on its own prose. The imports are parsed as an AST instead.
    """
    tree = ast.parse(Path(inspect.getfile(DecisionPolicy)).read_text())

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {
        "__future__",
        "math",
        "numbers",
        "dataclasses",
        "enum",
        "numpy",
        "pandas",
        "policy",
    }

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "open" not in called


# --- vectorised decisions -------------------------------------------------


def test_decide_many_agrees_with_the_scalar_decide():
    policy = make_policy()

    probabilities = pd.Series(
        [0.0, 0.1, 0.19999, 0.2, 0.45, 0.69999, 0.7, 0.95, 1.0]
    )

    vectorised = policy.decide_many(probabilities).tolist()
    scalar = [policy.decide(p).decision for p in probabilities]

    assert vectorised == scalar


def test_decide_many_preserves_index_and_row_alignment():
    policy = make_policy()

    probabilities = pd.Series(
        [0.05, 0.5, 0.95],
        index=["txn-c", "txn-a", "txn-b"],
    )

    decisions = policy.decide_many(probabilities)

    assert list(decisions.index) == ["txn-c", "txn-a", "txn-b"]
    assert decisions.name == DECISION_COLUMN_NAME
    assert decisions.tolist() == [Decision.ALLOW, Decision.REVIEW, Decision.BLOCK]


def test_decide_many_covers_every_row():
    policy = make_policy()

    probabilities = pd.Series(np.linspace(0.0, 1.0, 1_000))

    decisions = policy.decide_many(probabilities)

    assert len(decisions) == 1_000
    assert set(decisions.unique()) <= set(Decision)
    assert decisions.value_counts().sum() == 1_000


def test_decide_many_rejects_out_of_range_and_missing_values():
    policy = make_policy()

    with pytest.raises(ValueError, match="between 0 and 1"):
        policy.decide_many(pd.Series([0.1, 1.5]))

    with pytest.raises(ValueError, match="non-finite"):
        policy.decide_many(pd.Series([0.1, np.nan]))


def test_decide_many_rejects_a_dataframe():
    with pytest.raises(TypeError, match="pandas Series"):
        make_policy().decide_many(pd.DataFrame({"p": [0.1, 0.9]}))


# --- STEP 12 — layer independence -----------------------------------------


def test_the_policy_package_does_not_import_the_ml_package():
    """The model predicts risk; the policy decides. They must not couple."""
    policy_dir = Path(inspect.getfile(DecisionPolicy)).parent

    sources = sorted(policy_dir.glob("*.py"))

    assert sources, "no policy sources found"

    forbidden = {"ml", "xgboost", "sklearn", "scipy"}

    for path in sources:
        tree = ast.parse(path.read_text())

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        assert not (imported & forbidden), path.name


def test_decide_takes_only_a_probability():
    """No features, no frame, no estimator can reach the decision."""
    parameters = list(inspect.signature(DecisionPolicy.decide).parameters)

    assert parameters == ["self", "risk_probability"]
