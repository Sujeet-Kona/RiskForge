from pathlib import Path

import pandas as pd


FEATURE_PATH = Path("data/processed/train_behavioral_v1.parquet")
TRANSACTION_PATH = Path("data/raw/ieee_cis/train_transaction.csv")


FEATURES = [
    "card_txn_count_1h",
    "card_txn_count_24h",
    "card_amount_sum_24h",
    "address_txn_count_1h",
    "address_txn_count_24h",
    "address_amount_sum_24h",
]


def fraud_rate(df):
    return df["isFraud"].mean() * 100


def summarize(df, feature):
    print("\n" + "=" * 80)
    print(feature)
    print("=" * 80)

    value = df[feature]

    groups = pd.Series(
        "history_present",
        index=df.index,
        dtype="object",
    )

    groups[value == 0] = "zero"
    groups[(value > 0) & (value <= 1)] = "1"
    groups[(value > 1) & (value <= 5)] = "2_to_5"
    groups[(value > 5) & (value <= 20)] = "6_to_20"
    groups[value > 20] = "21_plus"

    result = (
        df.assign(_group=groups)
        .groupby("_group", observed=False)
        .agg(
            rows=("isFraud", "size"),
            fraud_rate_pct=("isFraud", "mean"),
        )
        .reset_index()
    )

    result["fraud_rate_pct"] *= 100

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def main():
    print("=" * 80)
    print("RISKFORGE — BEHAVIORAL HISTORY EFFECT AUDIT")
    print("=" * 80)

    print("\n[1/3] Loading features...")
    features = pd.read_parquet(FEATURE_PATH)

    print(f"Feature rows: {len(features):,}")

    print("\n[2/3] Loading fraud labels...")
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

    print(f"Joined rows: {len(df):,}")
    print(f"Overall fraud rate: {fraud_rate(df):.3f}%")

    print("\n[3/3] History effect analysis...")

    for feature in FEATURES:
        summarize(df, feature)

    print("\n" + "=" * 80)
    print("HISTORY EFFECT AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
