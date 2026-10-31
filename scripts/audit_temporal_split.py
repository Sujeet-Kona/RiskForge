from __future__ import annotations

import pandas as pd

from ml.split.temporal import temporal_split


INPUT_PATH = "data/raw/ieee_cis/train_transaction.csv"


def fraud_rate(labels: pd.Series) -> float:
    return float(labels.mean() * 100)


def main() -> None:
    print("=" * 80)
    print("RISKFORGE — CHRONOLOGICAL SPLIT AUDIT")
    print("=" * 80)

    print("\n[1/4] Loading transaction data...")

    df = pd.read_csv(
        INPUT_PATH,
        usecols=[
            "TransactionID",
            "TransactionDT",
            "isFraud",
        ],
    )

    print(f"Rows loaded: {len(df):,}")

    if not df["TransactionDT"].is_monotonic_increasing:
        df = df.sort_values("TransactionDT").reset_index(drop=True)

    print("\n[2/4] Applying chronological 70/15/15 split...")

    result = temporal_split(df)

    print(
        f"Train      : {len(result.train):,} rows "
        f"({len(result.train) / len(df) * 100:.2f}%)"
    )
    print(
        f"Validation : {len(result.validation):,} rows "
        f"({len(result.validation) / len(df) * 100:.2f}%)"
    )
    print(
        f"Test       : {len(result.test):,} rows "
        f"({len(result.test) / len(df) * 100:.2f}%)"
    )

    print("\n[3/4] Fraud-rate distribution")
    print("-" * 80)

    for name, part in [
        ("TRAIN", result.train),
        ("VALIDATION", result.validation),
        ("TEST", result.test),
    ]:
        print(
            f"{name:12s} "
            f"rows={len(part):8,} "
            f"fraud={int(part['isFraud'].sum()):7,} "
            f"fraud_rate={fraud_rate(part['isFraud']):.3f}% "
            f"time=[{part['TransactionDT'].min():,}, "
            f"{part['TransactionDT'].max():,}]"
        )

    print("\n[4/4] Leakage / boundary checks")
    print("-" * 80)

    train_max = result.train["TransactionDT"].max()
    validation_min = result.validation["TransactionDT"].min()
    validation_max = result.validation["TransactionDT"].max()
    test_min = result.test["TransactionDT"].min()

    assert train_max < validation_min
    assert validation_max < test_min

    train_times = set(result.train["TransactionDT"])
    validation_times = set(result.validation["TransactionDT"])
    test_times = set(result.test["TransactionDT"])

    assert train_times.isdisjoint(validation_times)
    assert train_times.isdisjoint(test_times)
    assert validation_times.isdisjoint(test_times)

    total_rows = (
        len(result.train)
        + len(result.validation)
        + len(result.test)
    )

    assert total_rows == len(df)

    total_ids = (
        set(result.train["TransactionID"])
        | set(result.validation["TransactionID"])
        | set(result.test["TransactionID"])
    )

    assert len(total_ids) == len(df)

    print("Chronological ordering       : PASS")
    print("Timestamp groups isolated    : PASS")
    print("All rows preserved            : PASS")
    print("Transaction IDs disjoint      : PASS")

    print("\n" + "=" * 80)
    print("TEMPORAL SPLIT AUDIT PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
