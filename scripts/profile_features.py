from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/ieee_cis")

TRANSACTION_PATH = RAW_DIR / "train_transaction.csv"
IDENTITY_PATH = RAW_DIR / "train_identity.csv"


def profile_file(path: Path) -> pd.DataFrame:
    print("=" * 100)
    print(f"FILE: {path.name}")
    print("=" * 100)

    df = pd.read_csv(path)

    profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_pct": (
                df.isna().mean().values * 100
            ),
            "unique_count": [
                df[column].nunique(dropna=True)
                for column in df.columns
            ],
        }
    )

    profile["non_missing_pct"] = 100 - profile["missing_pct"]

    profile = profile[
        [
            "column",
            "dtype",
            "missing_pct",
            "non_missing_pct",
            "unique_count",
        ]
    ]

    print("\nColumn profile:")
    print(profile.to_string(index=False))

    print("\nSummary:")
    print(f"Columns: {len(profile):,}")
    print(
        f"Columns with >50% missing: "
        f"{(profile['missing_pct'] > 50).sum():,}"
    )
    print(
        f"Columns with >90% missing: "
        f"{(profile['missing_pct'] > 90).sum():,}"
    )

    return profile


def main() -> None:
    transaction_profile = profile_file(TRANSACTION_PATH)
    identity_profile = profile_file(IDENTITY_PATH)

    output_dir = Path("ml/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    transaction_profile.to_csv(
        output_dir / "transaction_feature_profile.csv",
        index=False,
    )

    identity_profile.to_csv(
        output_dir / "identity_feature_profile.csv",
        index=False,
    )

    print("\nSaved:")
    print("  ml/data/transaction_feature_profile.csv")
    print("  ml/data/identity_feature_profile.csv")


if __name__ == "__main__":
    main()