from __future__ import annotations

import pandas as pd
import pytest

from ml.models.evaluation import (
    evaluate_predictions,
    metrics_to_dict,
)


def test_evaluate_predictions_returns_expected_confusion_matrix() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.2, 0.8, 0.9])

    metrics = evaluate_predictions(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert metrics.true_negatives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.true_positives == 2
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_metrics_to_dict_contains_all_metrics() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.7, 0.8, 0.2])

    metrics = evaluate_predictions(
        y_true,
        probabilities,
        threshold=0.5,
    )

    result = metrics_to_dict(metrics)

    expected = {
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    }

    assert set(result) == expected


def test_rejects_invalid_threshold() -> None:
    y_true = pd.Series([0, 1])
    probabilities = pd.Series([0.2, 0.8])

    with pytest.raises(ValueError, match="threshold"):
        evaluate_predictions(
            y_true,
            probabilities,
            threshold=1.0,
        )


def test_rejects_probability_outside_range() -> None:
    y_true = pd.Series([0, 1])
    probabilities = pd.Series([0.2, 1.2])

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        evaluate_predictions(
            y_true,
            probabilities,
        )


def test_rejects_length_mismatch() -> None:
    y_true = pd.Series([0, 1])
    probabilities = pd.Series([0.2])

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        evaluate_predictions(
            y_true,
            probabilities,
        )
