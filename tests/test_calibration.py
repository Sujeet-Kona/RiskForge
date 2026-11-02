import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.models.calibration import (
    CALIBRATION_METHODS,
    CALIBRATED_PROBABILITY_NAME,
    DEFAULT_CALIBRATION_FOLDS,
    apply_calibrator,
    build_calibrator,
    calibration_metrics_to_dict,
    cross_validate_calibration,
    evaluate_calibration,
    expected_calibration_error,
    fit_calibrator,
    reliability_table,
    temporal_fold_assignments,
)


def make_scores(n: int = 400, seed: int = 7) -> tuple[pd.Series, pd.Series]:
    """An inflated, miscalibrated score series plus its true labels.

    Mimics the locked XGBoost model: scores are pushed far above the event
    rate by ``scale_pos_weight``, so the raw probabilities are wrong in level
    while still ranking well.
    """
    rng = np.random.default_rng(seed)

    labels = pd.Series(
        (rng.random(n) < 0.05).astype(int),
        name="isFraud",
    )

    signal = labels.to_numpy() * 1.6 + rng.normal(0.0, 1.0, n)
    scores = 1.0 / (1.0 + np.exp(-(signal - 1.2)))

    return pd.Series(scores, name="fraud_probability"), labels


def make_groups(n: int, seed: int = 7) -> pd.Series:
    """Ascending timestamps with repeated values, as the real data has."""
    rng = np.random.default_rng(seed)

    steps = rng.integers(0, 3, n)
    values = np.cumsum(steps) + 86_400

    return pd.Series(values, name="TransactionDT")


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibrator_cannot_be_used_before_fitting(method):
    calibrator = build_calibrator(method)

    assert calibrator.estimator is None

    scores, _ = make_scores(20)

    with pytest.raises(NotFittedError):
        apply_calibrator(calibrator, scores)


def test_build_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="Unknown calibration method"):
        build_calibrator("platt-isotonic-hybrid")


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_fitting_produces_a_usable_estimator(method):
    scores, labels = make_scores()

    calibrator = fit_calibrator(method, scores, labels)

    assert calibrator.method == method
    assert calibrator.estimator is not None


def test_isotonic_uses_a_bounded_clipping_estimator():
    scores, labels = make_scores()

    calibrator = fit_calibrator("isotonic", scores, labels)

    assert isinstance(calibrator.estimator, IsotonicRegression)
    assert calibrator.estimator.y_min == 0.0
    assert calibrator.estimator.y_max == 1.0
    assert calibrator.estimator.out_of_bounds == "clip"


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibrated_probabilities_stay_within_the_unit_interval(method):
    scores, labels = make_scores()

    calibrator = fit_calibrator(method, scores, labels)
    calibrated = apply_calibrator(calibrator, scores)

    assert ((calibrated >= 0.0) & (calibrated <= 1.0)).all()
    assert calibrated.name == CALIBRATED_PROBABILITY_NAME


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibrated_probabilities_contain_no_nan_or_inf(method):
    scores, labels = make_scores()

    calibrator = fit_calibrator(method, scores, labels)
    calibrated = apply_calibrator(calibrator, scores).to_numpy(dtype=float)

    assert not np.isnan(calibrated).any()
    assert np.isfinite(calibrated).all()


def test_calibration_preserves_row_alignment_on_a_shuffled_index():
    scores, labels = make_scores(120)

    shuffled = [57, 3, 99, 0, 118, 42, 7, 81, 15, 63]
    reordered = scores.iloc[shuffled]
    reordered.index = [f"txn-{i}" for i in range(len(reordered))]

    calibrator = fit_calibrator("sigmoid", scores, labels)
    calibrated = apply_calibrator(calibrator, reordered)

    assert list(calibrated.index) == list(reordered.index)
    assert len(calibrated) == len(reordered)

    expected = apply_calibrator(calibrator, scores).loc[
        scores.iloc[shuffled].index
    ]
    np.testing.assert_allclose(
        calibrated.to_numpy(),
        expected.to_numpy(),
    )


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_the_mapping_is_learned_from_the_fit_labels(method):
    """Inverting the fit labels must invert the mapping, not pass it through."""
    scores, labels = make_scores(300)

    fit_scores = scores.iloc[:200]
    fit_labels = labels.iloc[:200]
    held_out_scores = scores.iloc[200:]

    baseline = apply_calibrator(
        fit_calibrator(method, fit_scores, fit_labels),
        held_out_scores,
    ).to_numpy()

    inverted = apply_calibrator(
        fit_calibrator(method, fit_scores, 1 - fit_labels),
        held_out_scores,
    ).to_numpy()

    assert not np.allclose(baseline, inverted)
    assert baseline.mean() < inverted.mean()


def test_no_calibration_entry_point_accepts_a_held_out_partition():
    """Structural guard: there is no parameter through which TEST can enter."""
    for function in (fit_calibrator, cross_validate_calibration):
        parameters = list(inspect.signature(function).parameters)

        assert "test" not in parameters
        assert not any("test" in name for name in parameters)


def test_cross_validation_does_not_leak_the_held_out_fold_labels():
    """A fold's own labels must have no effect on its own calibrated scores."""
    scores, labels = make_scores(400)
    groups = make_groups(400)

    folds = temporal_fold_assignments(groups, DEFAULT_CALIBRATION_FOLDS)
    target_fold = 0
    held_out = folds == target_fold

    honest = cross_validate_calibration(
        "isotonic",
        scores,
        labels,
        groups,
        DEFAULT_CALIBRATION_FOLDS,
    )

    tampered_labels = labels.copy()
    tampered_labels[held_out] = 1 - tampered_labels[held_out]

    tampered = cross_validate_calibration(
        "isotonic",
        scores,
        tampered_labels,
        groups,
        DEFAULT_CALIBRATION_FOLDS,
    )

    np.testing.assert_allclose(
        honest.to_numpy()[held_out],
        tampered.to_numpy()[held_out],
    )


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_cross_validation_covers_every_row(method):
    scores, labels = make_scores(300)
    groups = make_groups(300)

    out_of_fold = cross_validate_calibration(
        method,
        scores,
        labels,
        groups,
        DEFAULT_CALIBRATION_FOLDS,
    )

    assert len(out_of_fold) == len(scores)
    assert list(out_of_fold.index) == list(scores.index)
    assert out_of_fold.notna().all()
    assert ((out_of_fold >= 0.0) & (out_of_fold <= 1.0)).all()


def test_temporal_folds_keep_timestamp_groups_intact_and_ordered():
    groups = pd.Series(
        [100, 100, 100, 200, 200, 300, 400, 400, 500, 600],
        name="TransactionDT",
    )

    folds = temporal_fold_assignments(groups, 5)

    frame = pd.DataFrame({"group": groups, "fold": folds})

    assert frame.groupby("group")["fold"].nunique().eq(1).all()
    assert folds.min() == 0
    assert folds.max() == 4

    ordered = frame.drop_duplicates("group")
    assert ordered["fold"].is_monotonic_increasing


def test_temporal_folds_reject_unsorted_groups():
    with pytest.raises(ValueError, match="ascending order"):
        temporal_fold_assignments(pd.Series([3, 1, 2]), 2)


def test_temporal_folds_reject_too_few_groups():
    with pytest.raises(ValueError, match="Not enough distinct groups"):
        temporal_fold_assignments(pd.Series([1, 1, 1, 1]), 3)


def test_sigmoid_calibration_preserves_ranking_exactly():
    scores, labels = make_scores()

    calibrated = apply_calibrator(
        fit_calibrator("sigmoid", scores, labels),
        scores,
    )

    assert average_precision_score(labels, calibrated) == pytest.approx(
        average_precision_score(labels, scores),
        abs=1e-12,
    )
    assert roc_auc_score(labels, calibrated) == pytest.approx(
        roc_auc_score(labels, scores),
        abs=1e-12,
    )


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibration_is_monotonic_in_the_raw_score(method):
    scores, labels = make_scores()

    calibrated = apply_calibrator(
        fit_calibrator(method, scores, labels),
        scores,
    )

    order = np.argsort(scores.to_numpy(), kind="stable")

    assert np.all(np.diff(calibrated.to_numpy()[order]) >= -1e-12)


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibration_moves_predictions_toward_the_observed_rate(method):
    scores, labels = make_scores()

    calibrated = apply_calibrator(
        fit_calibrator(method, scores, labels),
        scores,
    )

    raw_gap = abs(scores.mean() - labels.mean())
    calibrated_gap = abs(calibrated.mean() - labels.mean())

    assert calibrated_gap < raw_gap


def test_fit_rejects_mismatched_lengths():
    scores, labels = make_scores(50)

    with pytest.raises(ValueError, match="same length"):
        fit_calibrator("sigmoid", scores, labels.iloc[:40])


def test_fit_rejects_scores_outside_the_unit_interval():
    _, labels = make_scores(50)
    scores = pd.Series(np.linspace(-0.2, 1.4, 50))

    with pytest.raises(ValueError, match="between 0 and 1"):
        fit_calibrator("sigmoid", scores, labels)


def test_fit_rejects_missing_and_non_finite_scores():
    _, labels = make_scores(50)

    with_nan = pd.Series(np.linspace(0.0, 1.0, 50))
    with_nan.iloc[3] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        fit_calibrator("sigmoid", with_nan, labels)


def test_fit_rejects_a_single_class_partition():
    scores, _ = make_scores(50)
    labels = pd.Series(np.zeros(50, dtype=int))

    with pytest.raises(ValueError, match="both fraud and non-fraud"):
        fit_calibrator("sigmoid", scores, labels)


def test_fit_rejects_a_constant_raw_score():
    labels = pd.Series([0, 1, 0, 1])
    scores = pd.Series([0.5, 0.5, 0.5, 0.5])

    with pytest.raises(ValueError, match="two distinct raw scores"):
        fit_calibrator("isotonic", scores, labels)


def test_brier_and_log_loss_match_their_definitions():
    labels = pd.Series([1, 0])
    scores = pd.Series([0.5, 0.5])

    metrics = evaluate_calibration(labels, scores)

    assert metrics.brier_score == pytest.approx(0.25)
    assert metrics.log_loss == pytest.approx(-np.log(0.5))
    assert metrics.n_samples == 2


def test_expected_calibration_error_is_zero_when_predictions_are_exact():
    labels = pd.Series([1, 0, 1, 0])
    scores = pd.Series([0.5, 0.5, 0.5, 0.5])

    assert expected_calibration_error(labels, scores) == pytest.approx(0.0)


def test_expected_calibration_error_detects_a_systematically_inflated_score():
    labels = pd.Series([1, 0] * 50)
    inflated = pd.Series([0.95, 0.90] * 50)

    error = expected_calibration_error(labels, inflated)

    assert error > 0.4


def test_reliability_table_is_count_weighted_and_omits_empty_bins():
    scores, labels = make_scores()

    table = reliability_table(labels, scores, n_bins=10)

    assert sum(entry.count for entry in table) == len(scores)
    assert len(table) <= 10
    assert all(entry.count > 0 for entry in table)

    for entry in table:
        assert entry.lower < entry.upper
        assert entry.lower <= entry.mean_predicted_probability <= entry.upper
        assert 0.0 <= entry.observed_fraud_rate <= 1.0


def test_reliability_table_uses_meaningful_probability_bin_edges():
    labels = pd.Series([1, 0, 0, 1])
    scores = pd.Series([0.05, 0.25, 0.55, 0.95])

    table = reliability_table(labels, scores, n_bins=10)

    edges = [(round(e.lower, 10), round(e.upper, 10)) for e in table]

    assert edges == [(0.0, 0.1), (0.2, 0.3), (0.5, 0.6), (0.9, 1.0)]
    assert [entry.count for entry in table] == [1, 1, 1, 1]


def test_evaluate_calibration_reports_the_observed_rate_and_ranking():
    scores, labels = make_scores()

    metrics = evaluate_calibration(labels, scores)

    assert metrics.observed_fraud_rate == pytest.approx(labels.mean())
    assert metrics.pr_auc == pytest.approx(
        average_precision_score(labels, scores)
    )
    assert metrics.roc_auc == pytest.approx(roc_auc_score(labels, scores))


def test_evaluate_calibration_rejects_a_single_class_partition():
    scores, _ = make_scores(50)
    labels = pd.Series(np.ones(50, dtype=int))

    with pytest.raises(ValueError, match="both fraud and non-fraud"):
        evaluate_calibration(labels, scores)


def test_calibration_metrics_to_dict_is_serializable():
    scores, labels = make_scores()

    values = calibration_metrics_to_dict(
        evaluate_calibration(labels, scores)
    )

    assert set(values) == {
        "brier_score",
        "log_loss",
        "expected_calibration_error",
        "pr_auc",
        "roc_auc",
        "mean_predicted_probability",
        "observed_fraud_rate",
        "n_samples",
    }
    assert all(
        isinstance(value, (float, int)) for value in values.values()
    )
