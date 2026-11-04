from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


CALIBRATION_METHODS = ("sigmoid", "isotonic")

DEFAULT_CALIBRATION_FOLDS = 5
DEFAULT_RELIABILITY_BINS = 10

CALIBRATED_PROBABILITY_NAME = "calibrated_fraud_probability"

# Isotonic regression is free to return exactly 0.0 or 1.0, which makes the log
# loss unbounded. Clipping at machine epsilon keeps the metric finite without
# meaningfully changing its value.
_LOG_LOSS_EPSILON = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class FraudCalibrator:
    """A score-to-probability mapping fitted on held-out model scores.

    Instances are produced unfitted by :func:`build_calibrator` and must be
    passed through :func:`fit_calibrator` before use.
    """

    method: str
    estimator: Any | None = None


@dataclass(frozen=True)
class CalibrationMetrics:
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    pr_auc: float
    roc_auc: float
    mean_predicted_probability: float
    observed_fraud_rate: float
    n_samples: int


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    mean_predicted_probability: float
    observed_fraud_rate: float
    count: int


def _validate_scores(
    raw_probability: pd.Series,
    y_true: pd.Series | None,
) -> None:
    if raw_probability.isna().any():
        raise ValueError("raw_probability contains missing values.")

    if not np.isfinite(raw_probability.to_numpy(dtype=float)).all():
        raise ValueError("raw_probability contains non-finite values.")

    if ((raw_probability < 0) | (raw_probability > 1)).any():
        raise ValueError("raw_probability must be between 0 and 1.")

    if y_true is None:
        return

    if len(raw_probability) != len(y_true):
        raise ValueError(
            "raw_probability and y_true must have the same length."
        )

    if y_true.isna().any():
        raise ValueError("y_true contains missing values.")

    if not y_true.isin([0, 1]).all():
        raise ValueError("y_true must contain only 0/1 values.")


def _build_estimator(method: str) -> Any:
    if method == "sigmoid":
        # Effectively unregularised logistic regression on the raw score, which
        # is Platt scaling. No class weighting: the whole point is to recover
        # the true event rate, not a balanced decision surface.
        return LogisticRegression(
            C=1.0e10,
            solver="lbfgs",
            max_iter=1_000,
            random_state=42,
        )

    return IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )


def build_calibrator(method: str) -> FraudCalibrator:
    """Build an unfitted calibrator for one of :data:`CALIBRATION_METHODS`."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(
            f"Unknown calibration method {method!r}; "
            f"expected one of {list(CALIBRATION_METHODS)}."
        )

    return FraudCalibrator(method=method, estimator=None)


def fit_calibrator(
    method: str,
    raw_probability: pd.Series,
    y_true: pd.Series,
) -> FraudCalibrator:
    """Fit the raw-score-to-probability mapping.

    ``raw_probability`` must come from data the fraud model was not fitted on.
    Fitting on the model's own training scores teaches the calibrator an
    over-confident mapping, because those scores are in-sample.
    """
    # Rejects an unknown method up front; the fitted bundle is built below.
    build_calibrator(method)

    _validate_scores(raw_probability, y_true)

    if len(raw_probability) == 0:
        raise ValueError("Calibration requires at least one row.")

    if raw_probability.nunique() < 2:
        raise ValueError(
            "Calibration requires at least two distinct raw scores."
        )

    if y_true.nunique() < 2:
        raise ValueError(
            "Calibration requires both fraud and non-fraud cases."
        )

    estimator = _build_estimator(method)
    scores = raw_probability.to_numpy(dtype=float).reshape(-1, 1)
    labels = y_true.to_numpy(dtype=int)

    if method == "sigmoid":
        estimator.fit(scores, labels)
    else:
        estimator.fit(scores.ravel(), labels)

    return FraudCalibrator(method=method, estimator=estimator)


def apply_calibrator(
    calibrator: FraudCalibrator,
    raw_probability: pd.Series,
) -> pd.Series:
    """Map raw fraud scores to calibrated fraud probabilities."""
    if calibrator.estimator is None:
        raise NotFittedError(
            f"{calibrator.method} calibrator must be fitted before use."
        )

    _validate_scores(raw_probability, None)

    scores = raw_probability.to_numpy(dtype=float)

    if calibrator.method == "sigmoid":
        calibrated = calibrator.estimator.predict_proba(
            scores.reshape(-1, 1)
        )[:, 1]
    else:
        calibrated = calibrator.estimator.predict(scores)

    calibrated = np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)

    return pd.Series(
        calibrated,
        index=raw_probability.index,
        name=CALIBRATED_PROBABILITY_NAME,
    )


def temporal_fold_assignments(
    groups: pd.Series,
    n_folds: int = DEFAULT_CALIBRATION_FOLDS,
) -> np.ndarray:
    """Assign each row a cross-validation fold in chronological order.

    Rows sharing a timestamp are always placed in the same fold, mirroring the
    invariant the train/validation/test split already enforces.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    if groups.isna().any():
        raise ValueError("groups must not contain missing values.")

    if not pd.Series(groups).is_monotonic_increasing:
        raise ValueError("groups must be sorted in ascending order.")

    unique_groups = groups.drop_duplicates()

    if len(unique_groups) < n_folds:
        raise ValueError(
            "Not enough distinct groups to form the requested folds."
        )

    position = pd.Series(
        np.arange(len(unique_groups)),
        index=unique_groups.to_numpy(),
    )

    group_index = groups.map(position).to_numpy()

    fold = np.floor(
        group_index * n_folds / len(unique_groups)
    ).astype(int)

    return np.minimum(fold, n_folds - 1)


def cross_validate_calibration(
    method: str,
    raw_probability: pd.Series,
    y_true: pd.Series,
    groups: pd.Series,
    n_folds: int = DEFAULT_CALIBRATION_FOLDS,
) -> pd.Series:
    """Return out-of-fold calibrated scores for honest method selection.

    The first chronological fold is an unscored warm-up period. Each later
    fold is scored by a calibrator fitted only on strictly earlier folds.
    Predictions for earlier periods can therefore never depend on later labels.
    The returned series contains NaN for the warm-up rows; evaluate only rows
    with a finite out-of-fold prediction.
    """
    _validate_scores(raw_probability, y_true)

    if len(groups) != len(raw_probability):
        raise ValueError("groups and raw_probability must have the same length.")

    if not y_true.index.equals(raw_probability.index):
        raise ValueError("y_true and raw_probability must have matching indexes.")

    if not groups.index.equals(raw_probability.index):
        raise ValueError("groups and raw_probability must have matching indexes.")

    folds = temporal_fold_assignments(groups, n_folds)

    # The first chronological fold is a warm-up period. It cannot be scored
    # honestly because no earlier calibration labels exist. Every later fold
    # is calibrated using only strictly earlier folds (expanding-window CV).
    out_of_fold = np.full(len(raw_probability), np.nan)

    for fold in range(1, n_folds):
        fit_rows = folds < fold
        score_rows = folds == fold

        if not score_rows.any():
            raise ValueError(f"Cross-validation fold {fold} is empty.")

        fit_labels = y_true.iloc[fit_rows]
        fit_scores = raw_probability.iloc[fit_rows]

        if fit_labels.nunique() < 2:
            raise ValueError(
                f"Cross-validation fold {fold} has a single-class "
                "strictly-prior calibration fit set."
            )

        if fit_scores.nunique() < 2:
            raise ValueError(
                f"Cross-validation fold {fold} has fewer than two distinct "
                "strictly-prior calibration scores."
            )

        calibrator = fit_calibrator(
            method,
            fit_scores,
            fit_labels,
        )

        out_of_fold[score_rows] = apply_calibrator(
            calibrator,
            raw_probability.iloc[score_rows],
        ).to_numpy()

    return pd.Series(
        out_of_fold,
        index=raw_probability.index,
        name=CALIBRATED_PROBABILITY_NAME,
    )


def reliability_table(
    y_true: pd.Series,
    probability: pd.Series,
    n_bins: int = DEFAULT_RELIABILITY_BINS,
) -> list[ReliabilityBin]:
    """Group predictions into equal-width probability bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    _validate_scores(probability, y_true)

    values = probability.to_numpy(dtype=float)
    labels = y_true.to_numpy(dtype=float)

    # Integer division rather than linspace, so edges like 0.3 are exact.
    edges = np.arange(n_bins + 1, dtype=float) / n_bins

    bins = np.minimum((values * n_bins).astype(int), n_bins - 1)

    table: list[ReliabilityBin] = []

    for index in range(n_bins):
        members = bins == index
        count = int(members.sum())

        if count == 0:
            continue

        table.append(
            ReliabilityBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                mean_predicted_probability=float(values[members].mean()),
                observed_fraud_rate=float(labels[members].mean()),
                count=count,
            )
        )

    return table


def expected_calibration_error(
    y_true: pd.Series,
    probability: pd.Series,
    n_bins: int = DEFAULT_RELIABILITY_BINS,
) -> float:
    """Count-weighted mean absolute gap between predicted and observed rate."""
    table = reliability_table(y_true, probability, n_bins=n_bins)

    total = sum(entry.count for entry in table)

    if total == 0:
        raise ValueError("Expected calibration error requires at least one row.")

    return float(
        sum(
            (entry.count / total)
            * abs(
                entry.mean_predicted_probability - entry.observed_fraud_rate
            )
            for entry in table
        )
    )


def evaluate_calibration(
    y_true: pd.Series,
    probability: pd.Series,
    n_bins: int = DEFAULT_RELIABILITY_BINS,
) -> CalibrationMetrics:
    """Report calibration quality alongside the retained ranking metrics."""
    _validate_scores(probability, y_true)

    values = probability.to_numpy(dtype=float)
    labels = y_true.to_numpy(dtype=int)

    if len(values) == 0:
        raise ValueError("Calibration evaluation requires at least one row.")

    if y_true.nunique() < 2:
        raise ValueError(
            "Calibration evaluation requires both fraud and non-fraud cases."
        )

    clipped = np.clip(values, _LOG_LOSS_EPSILON, 1.0 - _LOG_LOSS_EPSILON)

    return CalibrationMetrics(
        brier_score=float(np.mean((values - labels) ** 2)),
        log_loss=float(log_loss(labels, clipped, labels=[0, 1])),
        expected_calibration_error=expected_calibration_error(
            y_true,
            probability,
            n_bins=n_bins,
        ),
        pr_auc=float(average_precision_score(labels, values)),
        roc_auc=float(roc_auc_score(labels, values)),
        mean_predicted_probability=float(values.mean()),
        observed_fraud_rate=float(labels.mean()),
        n_samples=int(len(values)),
    )


def calibration_metrics_to_dict(
    metrics: CalibrationMetrics,
) -> dict[str, float | int]:
    """Convert calibration metrics to a serializable dictionary."""
    return {
        "brier_score": metrics.brier_score,
        "log_loss": metrics.log_loss,
        "expected_calibration_error": metrics.expected_calibration_error,
        "pr_auc": metrics.pr_auc,
        "roc_auc": metrics.roc_auc,
        "mean_predicted_probability": metrics.mean_predicted_probability,
        "observed_fraud_rate": metrics.observed_fraud_rate,
        "n_samples": metrics.n_samples,
    }
