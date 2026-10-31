from pathlib import Path
import time

import pandas as pd

from ml.features.behavioral import build_behavioral_features


RAW_PATH = Path("data/raw/ieee_cis/train_transaction.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "train_behavioral_v1.parquet"

INPUT_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "addr1",
    "addr2",
]


def main() -> None:
    print("=" * 80)
    print("RISKFORGE — BUILD BEHAVIORAL FEATURES V1")
    print("=" * 80)

    start = time.perf_counter()

    print("\n[1/4] Loading transaction data...")

    df = pd.read_csv(
        RAW_PATH,
        usecols=INPUT_COLUMNS,
    )

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    if not df["TransactionDT"].is_monotonic_increasing:
        raise ValueError(
            "TransactionDT must be monotonically increasing."
        )

    print("\n[2/4] Building point-in-time behavioral features...")

    result = build_behavioral_features(df)

    print(f"Rows generated: {len(result):,}")
    print(f"Columns generated: {len(result.columns)}")

    expected_features = [
        "transaction_hour_bucket",
        "transaction_day_index",
        "transaction_day_mod7",
        "amount_log1p",
        "card_txn_count_1h",
        "card_txn_count_24h",
        "card_amount_sum_24h",
        "card_has_history_1h",
        "card_has_history_24h",
        "address_txn_count_1h",
        "address_txn_count_24h",
        "address_amount_sum_24h",
        "address_has_history_1h",
        "address_has_history_24h",
    ]

    missing = [
        column
        for column in expected_features
        if column not in result.columns
    ]

    if missing:
        raise ValueError(
            f"Missing generated features: {missing}"
        )

    print("\n[3/4] Validating generated data...")

    if len(result) != len(df):
        raise ValueError("Output row count does not match input.")

    for column in expected_features:
        if result[column].isna().any():
            raise ValueError(
                f"Generated feature contains NaN values: {column}"
            )

    amount_features = [
        "card_amount_sum_24h",
        "address_amount_sum_24h",
    ]

    for column in amount_features:
        if (result[column] < 0).any():
            raise ValueError(
                f"Negative behavioral amount detected: {column}"
            )

    history_features = [
        "card_has_history_1h",
        "card_has_history_24h",
        "address_has_history_1h",
        "address_has_history_24h",
    ]

    for column in history_features:
        if result[column].dtype != bool:
            raise ValueError(
                f"History feature must have boolean dtype: {column}"
            )

    print("Generated feature validation passed.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[4/4] Writing Parquet output...")

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 80)
    print("FEATURE GENERATION COMPLETE")
    print("=" * 80)
    print(f"Output:      {OUTPUT_PATH}")
    print(f"Rows:        {len(result):,}")
    print(f"Columns:     {len(result.columns)}")
    print(f"Elapsed:     {elapsed:.2f}s")

    print("\nGenerated behavioral features:")
    for column in expected_features:
        print(f"  - {column}")


if __name__ == "__main__":
    main()
