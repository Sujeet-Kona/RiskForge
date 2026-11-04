from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/ieee_cis")
TRANSACTION_PATH = RAW_DIR / "train_transaction.csv"
IDENTITY_PATH = RAW_DIR / "train_identity.csv"


TRANSACTION_COLUMNS = [
    "TransactionDT",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]

IDENTITY_COLUMNS = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
    "id_12",
    "id_13",
    "id_14",
    "id_15",
    "id_16",
    "id_17",
    "id_18",
    "id_19",
    "id_20",
    "id_21",
    "id_22",
    "id_23",
    "id_24",
    "id_25",
    "id_26",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_32",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]


def profile_categorical_columns(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    print("=" * 100)
    print(name)
    print("=" * 100)

    for column in columns:
        series = df[column]

        non_null = series.dropna()

        if len(non_null) == 0:
            continue

        value_counts = non_null.value_counts()

        print(f"\n{column}")
        print(f"  unique: {non_null.nunique():,}")
        print(f"  missing: {series.isna().mean() * 100:.2f}%")
        print("  top 5 values:")

        for value, count in value_counts.head(5).items():
            percentage = count / len(non_null) * 100
            print(
                f"    {value!r}: "
                f"{count:,} ({percentage:.2f}%)"
            )


def main() -> None:
    transactions = pd.read_csv(
        TRANSACTION_PATH,
        usecols=TRANSACTION_COLUMNS,
    )

    identity = pd.read_csv(
        IDENTITY_PATH,
        usecols=IDENTITY_COLUMNS,
    )

    profile_categorical_columns(
        transactions,
        TRANSACTION_COLUMNS[1:],
        "TRANSACTION CATEGORICAL FEATURES",
    )

    profile_categorical_columns(
        identity,
        IDENTITY_COLUMNS[1:],
        "IDENTITY CATEGORICAL FEATURES",
    )


if __name__ == "__main__":
    main()