from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/ieee_cis")

TRANSACTION_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "isFraud",
]

IDENTITY_COLUMNS = [
    "TransactionID",
]


def main() -> None:
    transaction_path = RAW_DIR / "train_transaction.csv"
    identity_path = RAW_DIR / "train_identity.csv"

    transactions = pd.read_csv(
        transaction_path,
        usecols=TRANSACTION_COLUMNS,
    )

    identity = pd.read_csv(
        identity_path,
        usecols=IDENTITY_COLUMNS,
    )

    print("=" * 80)
    print("RISKFORGE TEMPORAL + JOIN AUDIT")
    print("=" * 80)

    print("\n1. Transaction ordering")
    is_sorted = transactions["TransactionDT"].is_monotonic_increasing
    print(f"TransactionDT monotonically increasing: {is_sorted}")

    print("\n2. TransactionDT")
    print(f"Minimum: {transactions['TransactionDT'].min():,}")
    print(f"Maximum: {transactions['TransactionDT'].max():,}")
    print(f"Unique values: {transactions['TransactionDT'].nunique():,}")

    print("\n3. Identity join coverage")

    identity_ids = set(identity["TransactionID"])

    transactions["has_identity"] = transactions["TransactionID"].isin(
        identity_ids
    )

    coverage = transactions["has_identity"].mean() * 100

    print(f"Transactions with identity: {coverage:.2f}%")
    print(
        f"Transactions without identity: "
        f"{100 - coverage:.2f}%"
    )

    print("\n4. Fraud rate by identity availability")

    fraud_by_identity = (
        transactions
        .groupby("has_identity")["isFraud"]
        .agg(["count", "sum", "mean"])
    )

    fraud_by_identity["mean"] *= 100

    print(fraud_by_identity.to_string())

    print("\n5. Fraud rate across chronological periods")

    transactions["time_bin"] = pd.qcut(
        transactions["TransactionDT"],
        q=10,
        duplicates="drop",
    )

    temporal_fraud = (
        transactions
        .groupby("time_bin", observed=True)["isFraud"]
        .agg(["count", "sum", "mean"])
    )

    temporal_fraud["mean"] *= 100

    print(temporal_fraud.to_string())


if __name__ == "__main__":
    main()