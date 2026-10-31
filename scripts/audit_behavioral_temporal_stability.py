from pathlib import Path

import pandas as pd


FEATURE_PATH = Path("data/processed/train_behavioral_v1.parquet")
TRANSACTION_PATH = Path("data/raw/ieee_cis/train_transaction.csv")


FEATURES = [
    "card_txn_count_1h",
    "card_txn_count_24h",
    "address_txn_count_1h",
    "address_txn_count_24h",
]


def main():
    print("=" * 80)
    print("RISKFORGE — BEHAVIORAL TEMPORAL STABILITY AUDIT")
    print("=" * 80)

    print("\n[1/4] Loading features...")
    features = pd.read_parquet(FEATURE_PATH)

    print("\n[2/4] Loading labels...")
    labels = pd.read_csv(
        TRANSACTION_PATH,
        usecols=["TransactionID", "isFraud"],
    )

    df = features.merge(
        labels,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    if df["isFraud"].isna().any():
        raise ValueError("Missing fraud labels after join.")

    # Chronological deciles.
    df["time_bin"] = pd.qcut(
        df["TransactionDT"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    print("\n[3/4] Overall fraud rate by chronological bin")
    print("-" * 80)

    overall = (
        df.groupby("time_bin", observed=True)
        .agg(
            rows=("isFraud", "size"),
            fraud_rate_pct=("isFraud", "mean"),
        )
        .reset_index()
    )

    overall["fraud_rate_pct"] *= 100

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\n[4/4] Zero-history fraud rate by chronological bin")
    print("-" * 80)

    for feature in FEATURES:
        print(f"\n### {feature}")

        summary = (
            df.assign(
                zero_history=df[feature] == 0,
            )
            .groupby(
                ["time_bin", "zero_history"],
                observed=True,
            )
            .agg(
                rows=("isFraud", "size"),
                fraud_rate_pct=("isFraud", "mean"),
            )
            .reset_index()
        )

        summary["fraud_rate_pct"] *= 100

        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    print("\n" + "=" * 80)
    print("TEMPORAL STABILITY AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
