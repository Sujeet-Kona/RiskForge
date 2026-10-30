from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/ieee_cis")
TRANSACTION_PATH = RAW_DIR / "train_transaction.csv"
IDENTITY_PATH = RAW_DIR / "train_identity.csv"


TRANSACTION_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "isFraud",
    "ProductCD",
    "card4",
    "card6",
    "card2",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
]

IDENTITY_COLUMNS = [
    "TransactionID",
    "DeviceType",
    "id_12",
    "id_15",
    "id_16",
    "id_28",
    "id_29",
    "DeviceInfo",
]


def fraud_rate_by_category(
    df: pd.DataFrame,
    column: str,
    min_count: int = 100,
) -> None:
    print("\n" + "=" * 100)
    print(f"FEATURE: {column}")
    print("=" * 100)

    grouped = (
        df.groupby(column, dropna=False)["isFraud"]
        .agg(
            transactions="count",
            fraud_count="sum",
            fraud_rate="mean",
        )
        .reset_index()
    )

    grouped["fraud_rate"] *= 100

    grouped = grouped[grouped["transactions"] >= min_count]

    grouped = grouped.sort_values(
        "fraud_rate",
        ascending=False,
    )

    print(grouped.to_string(index=False))


def missingness_fraud_rate(
    df: pd.DataFrame,
    column: str,
) -> None:
    print("\n" + "=" * 100)
    print(f"MISSINGNESS: {column}")
    print("=" * 100)

    df = df.copy()

    df["is_missing"] = df[column].isna()

    grouped = (
        df.groupby("is_missing")["isFraud"]
        .agg(
            transactions="count",
            fraud_count="sum",
            fraud_rate="mean",
        )
    )

    grouped["fraud_rate"] *= 100

    print(grouped.to_string())


def temporal_fraud_rate(
    df: pd.DataFrame,
    column: str,
    periods: int = 5,
) -> None:
    print("\n" + "=" * 100)
    print(f"TEMPORAL STABILITY: {column}")
    print("=" * 100)

    df = df.sort_values("TransactionDT").copy()

    df["time_period"] = pd.qcut(
        df["TransactionDT"],
        q=periods,
        labels=False,
        duplicates="drop",
    )

    grouped = (
        df.groupby(
            ["time_period", column],
            dropna=False,
        )["isFraud"]
        .agg(
            transactions="count",
            fraud_count="sum",
            fraud_rate="mean",
        )
        .reset_index()
    )

    grouped["fraud_rate"] *= 100

    grouped = grouped[grouped["transactions"] >= 100]

    print(grouped.to_string(index=False))


def main() -> None:
    transactions = pd.read_csv(
        TRANSACTION_PATH,
        usecols=TRANSACTION_COLUMNS,
    )

    identity = pd.read_csv(
        IDENTITY_PATH,
        usecols=IDENTITY_COLUMNS,
    )

    df = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    print("=" * 100)
    print("RISKFORGE FRAUD-SIGNAL AUDIT")
    print("=" * 100)

    categorical_features = [
        "ProductCD",
        "card4",
        "card6",
        "DeviceType",
        "id_12",
        "id_15",
        "id_16",
        "id_28",
        "id_29",
    ]

    missingness_features = [
        "card2",
        "addr1",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceInfo",
    ]

    for column in categorical_features:
        fraud_rate_by_category(df, column)

    for column in missingness_features:
        missingness_fraud_rate(df, column)

    for column in categorical_features:
        temporal_fraud_rate(df, column)


if __name__ == "__main__":
    main()