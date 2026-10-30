from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/ieee_cis")
TRANSACTION_FILE = DATA_DIR / "train_transaction.csv"


FEATURES = [
    "TransactionAmt",
    *[f"C{i}" for i in range(1, 15)],
    *[f"D{i}" for i in range(1, 16)],
]


def load_data() -> pd.DataFrame:
    columns = [
        "TransactionID",
        "TransactionDT",
        "isFraud",
        *FEATURES,
    ]

    return pd.read_csv(
        TRANSACTION_FILE,
        usecols=columns,
    )


def audit_feature(df: pd.DataFrame, feature: str) -> None:
    print("\n" + "=" * 80)
    print(f"FEATURE: {feature}")
    print("=" * 80)

    working = df[
        ["TransactionDT", "isFraud", feature]
    ].copy()

    working["missing"] = working[feature].isna()
    working["bin"] = "MISSING"

    non_missing = working.loc[
        ~working["missing"],
        feature,
    ]

    if non_missing.nunique() >= 10:
        try:
            bins = pd.qcut(
                non_missing,
                q=10,
                duplicates="drop",
            )
        except ValueError:
            bins = pd.cut(
                non_missing,
                bins=10,
            )

        working.loc[
            ~working["missing"],
            "bin",
        ] = bins.astype(str).to_numpy()

    else:
        working.loc[
            ~working["missing"],
            "bin",
        ] = non_missing.astype(str).to_numpy()

    summary = (
        working.groupby("bin", observed=False)
        .agg(
            transactions=("isFraud", "size"),
            fraud_count=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
        )
        .reset_index()
    )

    print("\nValue-bin fraud rates:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4%}",
        )
    )

    working["time_period"] = pd.qcut(
        working["TransactionDT"],
        q=5,
        labels=False,
        duplicates="drop",
    )

    print("\nTemporal fraud-rate range:")

    temporal = (
        working.groupby(
            ["time_period", "bin"],
            observed=False,
        )
        .agg(
            transactions=("isFraud", "size"),
            fraud_rate=("isFraud", "mean"),
        )
        .reset_index()
    )

    for bin_value in summary["bin"]:
        subset = temporal[
            temporal["bin"] == bin_value
        ]

        if len(subset) == 0:
            continue

        rates = subset["fraud_rate"]

        print(
            f"{str(bin_value):>25} | "
            f"min={rates.min():.4%} "
            f"max={rates.max():.4%}"
        )

def main() -> None:
    print("=" * 80)
    print("RISKFORGE — NUMERIC FRAUD SIGNAL AUDIT")
    print("=" * 80)

    df = load_data()

    print(f"\nRows: {len(df):,}")

    for feature in FEATURES:
        audit_feature(df, feature)


if __name__ == "__main__":
    main()