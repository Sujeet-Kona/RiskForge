from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.data.modeling import TARGET_COLUMN, build_modeling_frame
from ml.models.evaluation import evaluate_predictions, metrics_to_dict
from ml.models.xgboost_model import (
    fit_xgboost_model,
    predict_fraud_probability,
)
from ml.split.temporal import temporal_split


BEHAVIORAL_PATH = Path("data/processed/train_behavioral_v1.parquet")
TRANSACTION_PATH = Path("data/raw/ieee_cis/train_transaction.csv")


def print_metrics(name: str, metrics) -> None:
    print("\n" + "=" * 80)
    print(f"{name} METRICS")
    print("=" * 80)

    values = metrics_to_dict(metrics)

    for key, value in values.items():
        if isinstance(value, float):
            print(f"{key:24s}: {value:.6f}")
        else:
            print(f"{key:24s}: {value:,}")


def main() -> None:
    print("=" * 80)
    print("RISKFORGE — XGBOOST V1")
    print("=" * 80)

    print("\n[1/7] Loading behavioral artifact...")

    behavioral = pd.read_parquet(BEHAVIORAL_PATH)

    print(f"Behavioral rows: {len(behavioral):,}")

    print("\n[2/7] Loading fraud labels...")

    labels = pd.read_csv(
        TRANSACTION_PATH,
        usecols=["TransactionID", TARGET_COLUMN],
    )

    print(f"Label rows: {len(labels):,}")
    print(f"Fraud cases: {int(labels[TARGET_COLUMN].sum()):,}")
    print(f"Fraud rate: {labels[TARGET_COLUMN].mean():.3%}")

    print("\n[3/7] Building modeling frame...")

    modeling = build_modeling_frame(
        behavioral=behavioral,
        labels=labels,
    )

    print(f"Modeling rows: {len(modeling.data):,}")
    print(f"Model features: {len(modeling.feature_columns)}")

    print("\n[4/7] Applying chronological split...")

    split = temporal_split(modeling.data)

    for name, frame in [
        ("TRAIN", split.train),
        ("VALIDATION", split.validation),
        ("TEST", split.test),
    ]:
        fraud_count = int(frame[TARGET_COLUMN].sum())
        fraud_rate = frame[TARGET_COLUMN].mean()

        print(
            f"{name:10s} "
            f"rows={len(frame):8,} "
            f"fraud={fraud_count:6,} "
            f"fraud_rate={fraud_rate:.3%}"
        )

    print("\n[5/7] Training XGBoost...")

    model = fit_xgboost_model(split.train)

    print("Training complete.")
    print(f"scale_pos_weight: {model.scale_pos_weight:.6f}")

    print("\n[6/7] Generating fraud scores...")

    validation_probability = predict_fraud_probability(
        model,
        split.validation,
    )

    test_probability = predict_fraud_probability(
        model,
        split.test,
    )

    print("Score generation complete.")

    print("\n[7/7] Evaluating model...")

    validation_metrics = evaluate_predictions(
        split.validation[TARGET_COLUMN],
        validation_probability,
    )

    test_metrics = evaluate_predictions(
        split.test[TARGET_COLUMN],
        test_probability,
    )

    print_metrics("VALIDATION", validation_metrics)
    print_metrics("TEST", test_metrics)

    print("\n" + "=" * 80)
    print("XGBOOST V1 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
