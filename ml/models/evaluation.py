from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


def evaluate_predictions(
    y_true: pd.Series,
    fraud_probability: pd.Series,
    *,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Evaluate fraud probabilities at a reporting threshold."""

    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")

    if len(y_true) != len(fraud_probability):
        raise ValueError(
            "y_true and fraud_probability must have the same length."
        )

    if y_true.isna().any():
        raise ValueError("y_true contains missing values.")

    if fraud_probability.isna().any():
        raise ValueError(
            "fraud_probability contains missing values."
        )

    if not y_true.isin([0, 1]).all():
        raise ValueError("y_true must contain only 0/1 values.")

    if ((fraud_probability < 0) | (fraud_probability > 1)).any():
        raise ValueError(
            "fraud_probability must be between 0 and 1."
        )

    predicted = (fraud_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    ).ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return ClassificationMetrics(
        pr_auc=average_precision_score(
            y_true,
            fraud_probability,
        ),
        roc_auc=roc_auc_score(
            y_true,
            fraud_probability,
        ),
        precision=precision_score(
            y_true,
            predicted,
            zero_division=0,
        ),
        recall=recall_score(
            y_true,
            predicted,
            zero_division=0,
        ),
        f1=f1_score(
            y_true,
            predicted,
            zero_division=0,
        ),
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )


def metrics_to_dict(
    metrics: ClassificationMetrics,
) -> dict[str, float | int]:
    """Convert metrics to a serializable dictionary."""

    return {
        "pr_auc": metrics.pr_auc,
        "roc_auc": metrics.roc_auc,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "false_positive_rate": metrics.false_positive_rate,
        "false_negative_rate": metrics.false_negative_rate,
        "true_negatives": metrics.true_negatives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "true_positives": metrics.true_positives,
    }
