
import pandas as pd

TRANSACTION_PATH = "data/raw/ieee_cis/train_transaction.csv"

FEATURES = [
    "TransactionDT",
    "TransactionAmt",
    *[f"C{i}" for i in range(1, 15)],
    *[f"D{i}" for i in range(1, 16)],
]

df = pd.read_csv(
    TRANSACTION_PATH,
    usecols=FEATURES,
)

print("=" * 80)
print("RISKFORGE — NUMERIC FEATURE SEMANTICS AUDIT")
print("=" * 80)

print(f"\nRows: {len(df):,}")

for feature in FEATURES:
    s = df[feature]

    print("\n" + "=" * 80)
    print(f"FEATURE: {feature}")
    print("=" * 80)

    print(f"Missing: {s.isna().mean():.2%}")
    print(f"Unique:  {s.nunique(dropna=True):,}")

    if feature == "TransactionDT":
        print("Type: transaction time/order feature")
        print("Use: derive temporal features; do NOT use raw value directly.")

    elif feature == "TransactionAmt":
        print("Type: transaction amount")
        print("Use: candidate core feature.")

    elif feature.startswith("C"):
        print("Type: count-like / aggregated transaction feature")
        print("Leakage status: MUST be verified before final inclusion.")

    elif feature.startswith("D"):
        print("Type: time-delta / elapsed-time-like feature")
        print("Leakage status: MUST be verified before final inclusion.")

    print(f"Min:    {s.min()}")
    print(f"Median: {s.median()}")
    print(f"Max:    {s.max()}")
