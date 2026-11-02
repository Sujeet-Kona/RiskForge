from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    TARGET_COLUMN,
    build_modeling_frame,
)
from ml.models.evaluation import evaluate_predictions, metrics_to_dict
from ml.models.logistic_baseline import (
    fit_logistic_baseline,
    predict_fraud_probability,
)
from ml.split.temporal import temporal_split


BEHAVIORAL_PATH = Path(
    "data/processed/train_behavioral_v1.parquet"
)

TRANSACTION_PATH = Path(
    "data/raw/ieee_cis/train_transaction.csv"
)

TRANSACTION_COLUMNS = [
    "TransactionID",
    TARGET_COLUMN,
    *CATEGORICAL_FEATURES,
    *CD_FEATURES,
]


def print_metrics(
    name: str,
    metrics,
) -> None:
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    values = metrics_to_dict(metrics)

    for key, value in values.items():
        if isinstance(value, float):
            print(f"{key:24s}: {value:.6f}")
        else:
            print(f"{key:24s}: {value:,}")


def main() -> None:
    print("=" * 80)
    print("RISKFORGE — LOGISTIC REGRESSION BASELINE")
    print("=" * 80)

    print("\n[1/6] Loading behavioral artifact...")
    behavioral = pd.read_parquet(BEHAVIORAL_PATH)

    print(f"Behavioral rows: {len(behavioral):,}")

    print("\n[2/6] Loading transaction features and fraud labels...")
    transactions = pd.read_csv(
        TRANSACTION_PATH,
        usecols=TRANSACTION_COLUMNS,
    )

    print(f"Transaction rows: {len(transactions):,}")
    print(
        f"Fraud cases: {transactions[TARGET_COLUMN].sum():,}"
    )
    print(
        f"Fraud rate: {transactions[TARGET_COLUMN].mean():.3%}"
    )

    print("\n[3/6] Building modeling frame...")
    modeling = build_modeling_frame(
        behavioral=behavioral,
        transaction_features=transactions,
        labels=transactions,
    )

    print(
        f"Modeling rows: {len(modeling.data):,}"
    )
    print(
        f"Model features: {len(modeling.feature_columns)}"
    )

    print("\n[4/6] Applying chronological split...")
    split = temporal_split(modeling.data)

    for name, frame in [
        ("TRAIN", split.train),
        ("VALIDATION", split.validation),
        ("TEST", split.test),
    ]:
        print(
            f"{name:12s} "
            f"rows={len(frame):>8,} "
            f"fraud={frame['isFraud'].sum():>6,} "
            f"fraud_rate={frame['isFraud'].mean():.3%}"
        )

    print("\n[5/6] Training Logistic Regression...")
    model = fit_logistic_baseline(split.train)

    print("Training complete.")

    print("\n[6/6] Evaluating probabilities...")

    validation_probability = predict_fraud_probability(
        model,
        split.validation,
    )

    validation_metrics = evaluate_predictions(
        split.validation["isFraud"],
        validation_probability,
        threshold=0.5,
    )

    print_metrics(
        "VALIDATION METRICS",
        validation_metrics,
    )

    test_probability = predict_fraud_probability(
        model,
        split.test,
    )

    test_metrics = evaluate_predictions(
        split.test["isFraud"],
        test_probability,
        threshold=0.5,
    )

    print_metrics(
        "TEST METRICS",
        test_metrics,
    )

    print()
    print("=" * 80)
    print("LOGISTIC BASELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
