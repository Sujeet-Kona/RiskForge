from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TARGET_COLUMN = "isFraud"

MODEL_FEATURES = [
    "TransactionAmt",
    "amount_log1p",
    "transaction_hour_bucket",
    "transaction_day_mod7",
    "card_txn_count_1h",
    "card_txn_count_24h",
    "card_has_history_1h",
    "card_has_history_24h",
    "card_amount_sum_24h",
    "address_txn_count_1h",
    "address_txn_count_24h",
    "address_has_history_1h",
    "address_amount_sum_24h",
]

CATEGORICAL_FEATURES = [
    "transaction_hour_bucket",
    "transaction_day_mod7",
]

NUMERIC_FEATURES = [
    feature
    for feature in MODEL_FEATURES
    if feature not in CATEGORICAL_FEATURES
]

SPLIT_COLUMN = "TransactionDT"


@dataclass(frozen=True)
class ModelingFrame:
    data: pd.DataFrame
    feature_columns: list[str]
    categorical_features: list[str]
    numeric_features: list[str]


def build_modeling_frame(
    behavioral: pd.DataFrame,
    labels: pd.DataFrame,
) -> ModelingFrame:
    """Attach fraud labels and construct the first model feature contract.

    The raw behavioral entity identifiers remain available before this
    function but are deliberately excluded from MODEL_FEATURES.
    TransactionDT is retained only for chronological splitting.
    """

    if "TransactionID" not in behavioral.columns:
        raise ValueError(
            "Behavioral dataframe must contain TransactionID."
        )

    if "TransactionID" not in labels.columns:
        raise ValueError(
            "Labels dataframe must contain TransactionID."
        )

    if TARGET_COLUMN not in labels.columns:
        raise ValueError(
            f"Labels dataframe must contain {TARGET_COLUMN}."
        )

    required_behavioral = {
        "TransactionID",
        SPLIT_COLUMN,
        *MODEL_FEATURES,
    }

    missing_behavioral = sorted(
        required_behavioral - set(behavioral.columns)
    )

    if missing_behavioral:
        raise ValueError(
            "Behavioral dataframe is missing required columns: "
            f"{missing_behavioral}"
        )

    if labels["TransactionID"].duplicated().any():
        raise ValueError(
            "Labels contain duplicate TransactionID values."
        )

    if behavioral["TransactionID"].duplicated().any():
        raise ValueError(
            "Behavioral dataframe contains duplicate TransactionID values."
        )

    if labels[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"{TARGET_COLUMN} contains missing values."
        )

    if not labels[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(
            f"{TARGET_COLUMN} must contain only 0/1 values."
        )

    result = behavioral[
        ["TransactionID", SPLIT_COLUMN, *MODEL_FEATURES]
    ].merge(
        labels[["TransactionID", TARGET_COLUMN]],
        on="TransactionID",
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    if len(result) != len(behavioral):
        raise ValueError(
            "Label attachment changed the behavioral row count."
        )

    if result[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Label attachment produced missing targets."
        )

    if result["TransactionID"].duplicated().any():
        raise ValueError(
            "Modeling frame contains duplicate TransactionID values."
        )

    if not result[SPLIT_COLUMN].is_monotonic_increasing:
        raise ValueError(
            "Modeling frame must remain sorted by TransactionDT."
        )

    if result[MODEL_FEATURES].isna().any().any():
        missing_counts = (
            result[MODEL_FEATURES]
            .isna()
            .sum()
            .loc[lambda series: series > 0]
            .to_dict()
        )

        raise ValueError(
            "Model features contain missing values: "
            f"{missing_counts}"
        )

    return ModelingFrame(
        data=result,
        feature_columns=MODEL_FEATURES.copy(),
        categorical_features=CATEGORICAL_FEATURES.copy(),
        numeric_features=NUMERIC_FEATURES.copy(),
    )
