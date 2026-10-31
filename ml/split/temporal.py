from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_split(
    df: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Split data chronologically without separating equal timestamps."""

    if "TransactionDT" not in df.columns:
        raise ValueError("Input must contain TransactionDT.")

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1."
        )

    if df.empty:
        raise ValueError("Input dataframe must not be empty.")

    if df["TransactionDT"].isna().any():
        raise ValueError("TransactionDT must not contain missing values.")

    if not df["TransactionDT"].is_monotonic_increasing:
        raise ValueError(
            "Input must be sorted by TransactionDT in ascending order."
        )

    timestamps = df["TransactionDT"].drop_duplicates().tolist()

    train_cutoff = int(len(timestamps) * train_fraction)

    validation_cutoff = int(
        len(timestamps) * (train_fraction + validation_fraction)
    )

    if train_cutoff <= 0 or validation_cutoff <= train_cutoff:
        raise ValueError("Dataset is too small for the requested split.")

    train_end = timestamps[train_cutoff - 1]
    validation_end = timestamps[validation_cutoff - 1]

    train = df[df["TransactionDT"] <= train_end].copy()

    validation = df[
        (df["TransactionDT"] > train_end)
        & (df["TransactionDT"] <= validation_end)
    ].copy()

    test = df[df["TransactionDT"] > validation_end].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Temporal split produced an empty partition.")

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
    )
