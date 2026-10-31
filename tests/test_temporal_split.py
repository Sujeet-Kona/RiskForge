import pandas as pd
import pytest

from ml.split.temporal import temporal_split


def test_temporal_split_preserves_order():
    df = pd.DataFrame(
        {
            "TransactionDT": range(100),
            "value": range(100),
        }
    )

    result = temporal_split(df)

    assert result.train["TransactionDT"].max() < (
        result.validation["TransactionDT"].min()
    )

    assert result.validation["TransactionDT"].max() < (
        result.test["TransactionDT"].min()
    )


def test_same_timestamp_never_crosses_split():
    transaction_dt = [
        1, 1, 1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
        6,
    ]

    df = pd.DataFrame(
        {
            "TransactionDT": transaction_dt,
            "value": range(len(transaction_dt)),
        }
    )

    result = temporal_split(
        df,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    train_times = set(result.train["TransactionDT"])
    validation_times = set(result.validation["TransactionDT"])
    test_times = set(result.test["TransactionDT"])

    assert train_times.isdisjoint(validation_times)
    assert train_times.isdisjoint(test_times)
    assert validation_times.isdisjoint(test_times)

    all_times = train_times | validation_times | test_times

    assert all_times == set(transaction_dt)


def test_all_rows_are_preserved():
    df = pd.DataFrame(
        {
            "TransactionDT": range(100),
            "value": range(100),
        }
    )

    result = temporal_split(df)

    assert (
        len(result.train)
        + len(result.validation)
        + len(result.test)
        == len(df)
    )


def test_unsorted_input_is_rejected():
    df = pd.DataFrame(
        {
            "TransactionDT": [1, 3, 2],
        }
    )

    with pytest.raises(ValueError, match="sorted"):
        temporal_split(df)


def test_missing_transaction_dt_is_rejected():
    df = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(ValueError, match="TransactionDT"):
        temporal_split(df)
