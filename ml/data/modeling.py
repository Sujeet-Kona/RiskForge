from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TARGET_COLUMN = "isFraud"

SPLIT_COLUMN = "TransactionDT"

BEHAVIORAL_FEATURES = [
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
    "ProductCD",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
]

# D5 is deliberately absent: validation-only backward elimination showed that
# removing it improved validation PR-AUC (0.471864 -> 0.474352), and the locked
# test partition must not influence feature selection.
CD_FEATURES = [
    "C1",
    "C2",
    "C11",
    "C13",
    "D2",
    "D3",
    "D4",
    "D8",
    "D10",
    "D11",
    "D15",
]

MODEL_FEATURES = [
    *BEHAVIORAL_FEATURES,
    *CATEGORICAL_FEATURES,
    *CD_FEATURES,
]

NUMERIC_FEATURES = [
    *BEHAVIORAL_FEATURES,
    *CD_FEATURES,
]


@dataclass(frozen=True)
class ModelingFrame:
    data: pd.DataFrame
    feature_columns: list[str]
    categorical_features: list[str]
    numeric_features: list[str]


def build_modeling_frame(
    behavioral: pd.DataFrame,
    transaction_features: pd.DataFrame,
    labels: pd.DataFrame,
) -> ModelingFrame:
    """Assemble the official 28-feature modeling frame.

    ``behavioral`` supplies the point-in-time behavioral features plus
    ``TransactionDT``; ``transaction_features`` supplies the categorical
    transaction fields and the selected C/D benchmark features; ``labels``
    supplies the fraud target. A single raw transaction frame may be passed for
    both of the latter arguments, since each selects only the columns it needs.

    Entity identifiers (``card1``, ``addr1``, ``addr2``) stay out of
    MODEL_FEATURES, and ``TransactionDT`` is retained only for chronological
    splitting.

    Missingness is treated differently per feature family: behavioral features
    are computed for every row, so a NaN there means the feature pipeline
    failed and is rejected. C/D and categorical NaNs are meaningful upstream
    absences and are preserved for the preprocessing stage to encode.
    """

    if "TransactionID" not in behavioral.columns:
        raise ValueError(
            "Behavioral dataframe must contain TransactionID."
        )

    if "TransactionID" not in transaction_features.columns:
        raise ValueError(
            "Transaction feature dataframe must contain TransactionID."
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
        *BEHAVIORAL_FEATURES,
    }

    missing_behavioral = sorted(
        required_behavioral - set(behavioral.columns)
    )

    if missing_behavioral:
        raise ValueError(
            "Behavioral dataframe is missing required columns: "
            f"{missing_behavioral}"
        )

    required_transaction = {
        "TransactionID",
        *CATEGORICAL_FEATURES,
        *CD_FEATURES,
    }

    missing_transaction = sorted(
        required_transaction - set(transaction_features.columns)
    )

    if missing_transaction:
        raise ValueError(
            "Transaction feature dataframe is missing required columns: "
            f"{missing_transaction}"
        )

    if labels["TransactionID"].duplicated().any():
        raise ValueError(
            "Labels contain duplicate TransactionID values."
        )

    if transaction_features["TransactionID"].duplicated().any():
        raise ValueError(
            "Transaction features contain duplicate TransactionID values."
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
        ["TransactionID", SPLIT_COLUMN, *BEHAVIORAL_FEATURES]
    ].merge(
        transaction_features[
            ["TransactionID", *CATEGORICAL_FEATURES, *CD_FEATURES]
        ],
        on="TransactionID",
        how="inner",
        validate="one_to_one",
        sort=False,
    ).merge(
        labels[["TransactionID", TARGET_COLUMN]],
        on="TransactionID",
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    if len(result) != len(behavioral):
        raise ValueError(
            "Feature and label attachment changed the behavioral row count."
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

    if result[BEHAVIORAL_FEATURES].isna().any().any():
        missing_counts = (
            result[BEHAVIORAL_FEATURES]
            .isna()
            .sum()
            .loc[lambda series: series > 0]
            .to_dict()
        )

        raise ValueError(
            "Behavioral features contain missing values: "
            f"{missing_counts}"
        )

    return ModelingFrame(
        data=result,
        feature_columns=MODEL_FEATURES.copy(),
        categorical_features=CATEGORICAL_FEATURES.copy(),
        numeric_features=NUMERIC_FEATURES.copy(),
    )
