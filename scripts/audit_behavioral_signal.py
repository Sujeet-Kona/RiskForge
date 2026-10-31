from pathlib import Path

import pandas as pd


FEATURE_PATH = Path(
    "data/processed/train_behavioral_v1.parquet"
)

TRANSACTION_PATH = Path(
    "data/raw/ieee_cis/train_transaction.csv"
)


def fraud_rate_table(
    df: pd.DataFrame,
    feature: str,
    quantiles: int = 10,
) -> pd.DataFrame:
    """Calculate fraud rate across quantile buckets."""

    values = df[feature]

    if values.nunique() <= 1:
        return pd.DataFrame(
            {
                "bucket": ["constant"],
                "rows": [len(df)],
                "fraud_rate_pct": [
                    df["isFraud"].mean() * 100
                ],
            }
        )

    try:
        buckets = pd.qcut(
            values,
            q=quantiles,
            duplicates="drop",
        )
    except ValueError:
        buckets = values

    result = (
        df.assign(_bucket=buckets)
        .groupby("_bucket", observed=True)
        .agg(
            rows=("isFraud", "size"),
            fraud_rate_pct=("isFraud", "mean"),
        )
        .reset_index()
    )

    result["fraud_rate_pct"] *= 100

    return result


def history_coverage(
    df: pd.DataFrame,
    feature: str,
) -> dict[str, float]:
    """Measure how many transactions have prior history."""

    rows = len(df)
    with_history = (df[feature] > 0).sum()

    return {
        "rows": rows,
        "with_history_pct": (
            with_history / rows * 100
        ),
        "no_history_pct": (
            (rows - with_history) / rows * 100
        ),
    }


def main() -> None:
    print("=" * 80)
    print("RISKFORGE — BEHAVIORAL FRAUD SIGNAL AUDIT")
    print("=" * 80)

    print("\n[1/4] Loading generated features...")

    features = pd.read_parquet(FEATURE_PATH)

    print(f"Feature rows: {len(features):,}")

    print("\n[2/4] Loading fraud labels...")

    labels = pd.read_csv(
        TRANSACTION_PATH,
        usecols=["TransactionID", "isFraud"],
    )

    print(f"Label rows:   {len(labels):,}")

    if len(features) != len(labels):
        raise ValueError(
            "Feature and label row counts differ."
        )

    print("\n[3/4] Joining labels...")

    df = features.merge(
        labels,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    if df["isFraud"].isna().any():
        raise ValueError(
            "Some generated feature rows have no fraud label."
        )

    print(f"Joined rows: {len(df):,}")
    print(
        f"Overall fraud rate: "
        f"{df['isFraud'].mean() * 100:.3f}%"
    )

    behavioral_features = [
        "card_txn_count_1h",
        "card_txn_count_24h",
        "card_amount_sum_24h",
        "address_txn_count_1h",
        "address_txn_count_24h",
        "address_amount_sum_24h",
    ]

    print("\n[4/4] Signal analysis...")

    print("\n" + "-" * 80)
    print("HISTORY COVERAGE")
    print("-" * 80)

    for feature in [
        "card_txn_count_1h",
        "card_txn_count_24h",
        "address_txn_count_1h",
        "address_txn_count_24h",
    ]:
        coverage = history_coverage(df, feature)

        print(
            f"{feature:30s} "
            f"history={coverage['with_history_pct']:6.2f}% "
            f"no_history={coverage['no_history_pct']:6.2f}%"
        )

    print("\n" + "-" * 80)
    print("FRAUD RATE BY FEATURE DECILE")
    print("-" * 80)

    for feature in behavioral_features:
        print(f"\n### {feature}")

        table = fraud_rate_table(
            df,
            feature,
        )

        print(
            table.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    print("\n" + "=" * 80)
    print("BEHAVIORAL SIGNAL AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
