from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/raw/ieee_cis")
TRANSACTION_FILE = DATA_DIR / "train_transaction.csv"


def load_data() -> pd.DataFrame:
    transaction_features = [
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "TransactionAmt",
    ]

    transaction_features += [f"C{i}" for i in range(1, 15)]
    transaction_features += [f"D{i}" for i in range(1, 16)]

    return pd.read_csv(
        TRANSACTION_FILE,
        usecols=transaction_features,
    )


def audit_missingness(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("MISSINGNESS")
    print("=" * 80)

    numeric_features = [
        "TransactionAmt",
        *[f"C{i}" for i in range(1, 15)],
        *[f"D{i}" for i in range(1, 16)],
    ]

    rows = []

    for feature in numeric_features:
        missing_pct = df[feature].isna().mean() * 100

        rows.append(
            {
                "feature": feature,
                "missing_count": int(df[feature].isna().sum()),
                "missing_pct": missing_pct,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        "missing_pct",
        ascending=False,
    )

    print(result.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def audit_distributions(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("DISTRIBUTIONS")
    print("=" * 80)

    numeric_features = [
        "TransactionAmt",
        *[f"C{i}" for i in range(1, 15)],
        *[f"D{i}" for i in range(1, 16)],
    ]

    rows = []

    for feature in numeric_features:
        series = df[feature].dropna()

        rows.append(
            {
                "feature": feature,
                "unique": series.nunique(),
                "min": series.min(),
                "median": series.median(),
                "mean": series.mean(),
                "p95": series.quantile(0.95),
                "p99": series.quantile(0.99),
                "max": series.max(),
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def audit_fraud_relationship(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("FRAUD RELATIONSHIP")
    print("=" * 80)

    numeric_features = [
        "TransactionAmt",
        *[f"C{i}" for i in range(1, 15)],
        *[f"D{i}" for i in range(1, 16)],
    ]

    rows = []

    for feature in numeric_features:
        fraud = df.loc[df["isFraud"] == 1, feature].dropna()
        nonfraud = df.loc[df["isFraud"] == 0, feature].dropna()

        if len(fraud) == 0 or len(nonfraud) == 0:
            continue

        fraud_median = fraud.median()
        nonfraud_median = nonfraud.median()

        fraud_mean = fraud.mean()
        nonfraud_mean = nonfraud.mean()

        median_ratio = (
            fraud_median / nonfraud_median
            if nonfraud_median != 0
            else np.nan
        )

        rows.append(
            {
                "feature": feature,
                "fraud_median": fraud_median,
                "nonfraud_median": nonfraud_median,
                "median_ratio": median_ratio,
                "fraud_mean": fraud_mean,
                "nonfraud_mean": nonfraud_mean,
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.sort_values(
            "median_ratio",
            ascending=False,
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def audit_temporal_stability(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("TEMPORAL STABILITY")
    print("=" * 80)

    numeric_features = [
        "TransactionAmt",
        *[f"C{i}" for i in range(1, 15)],
        *[f"D{i}" for i in range(1, 16)],
    ]

    df = df.copy()

    df["time_period"] = pd.qcut(
        df["TransactionDT"],
        q=5,
        labels=False,
        duplicates="drop",
    )

    for feature in numeric_features:
        print(f"\n--- {feature} ---")

        result = (
            df.groupby("time_period", observed=True)[feature]
            .agg(
                count="count",
                median="median",
                mean="mean",
            )
        )

        print(
            result.to_string(
                float_format=lambda x: f"{x:.3f}",
            )
        )


def main() -> None:
    print("=" * 80)
    print("RISKFORGE — NUMERIC FEATURE AUDIT")
    print("=" * 80)

    df = load_data()

    print(f"\nRows: {len(df):,}")
    print(f"Columns audited: {len(df.columns):,}")

    audit_missingness(df)
    audit_distributions(df)
    audit_fraud_relationship(df)
    audit_temporal_stability(df)


if __name__ == "__main__":
    main()