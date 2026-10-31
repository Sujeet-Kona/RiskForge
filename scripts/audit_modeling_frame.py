from __future__ import annotations

import pandas as pd

from ml.data.modeling import (
    MODEL_FEATURES,
    build_modeling_frame,
)
from ml.split.temporal import temporal_split


BEHAVIORAL_PATH = "data/processed/train_behavioral_v1.parquet"
TRANSACTION_PATH = "data/raw/ieee_cis/train_transaction.csv"


def main() -> None:
    print("=" * 80)
    print("RISKFORGE — MODELING FRAME AUDIT")
    print("=" * 80)

    print("\n[1/6] Loading behavioral artifact...")
    behavioral = pd.read_parquet(BEHAVIORAL_PATH)

    print(f"Behavioral rows: {len(behavioral):,}")
    print(f"Behavioral columns: {len(behavioral.columns)}")

    print("\n[2/6] Loading fraud labels...")
    labels = pd.read_csv(
        TRANSACTION_PATH,
        usecols=["TransactionID", "isFraud"],
    )

    print(f"Label rows: {len(labels):,}")
    print(f"Fraud cases: {labels['isFraud'].sum():,}")
    print(
        f"Fraud rate: {labels['isFraud'].mean() * 100:.3f}%"
    )

    print("\n[3/6] Building modeling frame...")
    modeling = build_modeling_frame(
        behavioral,
        labels,
    )

    df = modeling.data

    print(f"Modeling rows: {len(df):,}")
    print(f"Model features: {len(modeling.feature_columns)}")

    print("\n[4/6] Contract checks")
    print("-" * 80)

    assert len(df) == len(behavioral)
    print("Row count preserved           : PASS")

    assert df["TransactionID"].is_unique
    print("Transaction IDs unique        : PASS")

    assert df["isFraud"].notna().all()
    print("Target has no missing values  : PASS")

    assert df["isFraud"].isin([0, 1]).all()
    print("Target values are 0/1         : PASS")

    assert not df[MODEL_FEATURES].isna().any().any()
    print("Model features have no NaNs   : PASS")

    assert df["TransactionDT"].is_monotonic_increasing
    print("TransactionDT remains sorted  : PASS")

    expected_features = set(MODEL_FEATURES)

    assert set(modeling.feature_columns) == expected_features
    print("Feature contract matches      : PASS")

    forbidden = {
        "TransactionID",
        "TransactionDT",
        "card1",
        "addr1",
        "addr2",
        "transaction_day_index",
    }

    assert not forbidden.intersection(modeling.feature_columns)
    print("Forbidden identifiers absent  : PASS")

    print("\n[5/6] Label reconciliation")
    print("-" * 80)

    original_fraud = int(labels["isFraud"].sum())
    modeling_fraud = int(df["isFraud"].sum())

    print(f"Original fraud cases : {original_fraud:,}")
    print(f"Modeling fraud cases : {modeling_fraud:,}")

    assert original_fraud == modeling_fraud
    print("Fraud labels preserved        : PASS")

    print("\n[6/6] Temporal split verification")
    print("-" * 80)

    split = temporal_split(df)

    for name, frame in [
        ("TRAIN", split.train),
        ("VALIDATION", split.validation),
        ("TEST", split.test),
    ]:
        print(
            f"{name:<12} "
            f"rows={len(frame):>8,} "
            f"fraud={int(frame['isFraud'].sum()):>6,} "
            f"fraud_rate={frame['isFraud'].mean() * 100:>6.3f}% "
            f"time=[{frame['TransactionDT'].min():,}, "
            f"{frame['TransactionDT'].max():,}]"
        )

    assert len(split.train) + len(split.validation) + len(split.test) == len(df)

    print("\n" + "=" * 80)
    print("MODELING FRAME AUDIT PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
