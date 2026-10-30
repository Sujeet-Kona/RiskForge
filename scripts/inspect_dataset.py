from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/ieee_cis")


def inspect_file(path: Path) -> None:
    print("=" * 80)
    print(f"FILE: {path.name}")
    print("=" * 80)

    df = pd.read_csv(path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nMemory usage:")
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"{memory_mb:.2f} MB")

    print("\nMissing values — top 20:")
    missing = df.isna().mean().sort_values(ascending=False)
    print((missing.head(20) * 100).round(2).to_string())

    if "TransactionID" in df.columns:
        print("\nTransactionID:")
        print(f"Unique: {df['TransactionID'].nunique():,}")
        print(f"Duplicates: {df['TransactionID'].duplicated().sum():,}")

    if "isFraud" in df.columns:
        print("\nTarget distribution:")
        print(df["isFraud"].value_counts(dropna=False).to_string())

        fraud_rate = df["isFraud"].mean() * 100
        print(f"Fraud rate: {fraud_rate:.4f}%")

    if "TransactionDT" in df.columns:
        print("\nTransactionDT:")
        print(f"Minimum: {df['TransactionDT'].min():,}")
        print(f"Maximum: {df['TransactionDT'].max():,}")

    if "TransactionAmt" in df.columns:
        print("\nTransactionAmt:")
        print(df["TransactionAmt"].describe().to_string())

    print()


def main() -> None:
    transaction_file = RAW_DIR / "train_transaction.csv"
    identity_file = RAW_DIR / "train_identity.csv"

    if not transaction_file.exists():
        raise FileNotFoundError(transaction_file)

    if not identity_file.exists():
        raise FileNotFoundError(identity_file)

    inspect_file(transaction_file)
    inspect_file(identity_file)


if __name__ == "__main__":
    main()