"""The serving feature frame, and the point-in-time rule it must obey.

Requirement 18 lives here: every behavioral feature of the scored transaction
has to be a function of strictly earlier rows only. The window edges are
asserted explicitly because an off-by-one at ``-3600`` or ``-86400`` would be a
silent disagreement with the offline pipeline rather than a crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    BEHAVIORAL_INPUT_COLUMNS,
    SERVING_DTYPES,
    build_serving_features,
)
from app.schemas import HistoryTransaction, TransactionPayload
from ml.data.modeling import CD_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES
from ml.features.behavioral import ONE_DAY, ONE_HOUR, build_behavioral_features


CURRENT_DT = 1_000_000


def _transaction(**overrides) -> TransactionPayload:
    values = {
        "TransactionID": "3000001",
        "TransactionDT": CURRENT_DT,
        "TransactionAmt": 117.5,
        "card1": 13926,
        "addr1": 299.0,
        "addr2": 87.0,
        "ProductCD": "W",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": None,
        "C1": 1.0,
        "C2": 1.0,
        "C11": 2.0,
        "C13": 1.0,
        "D4": 0.0,
        "D10": 1.0,
        "D11": 1.0,
        "D15": 0.0,
    }
    values.update(overrides)

    return TransactionPayload(**values)


def _history(transaction_dt: int, **overrides) -> HistoryTransaction:
    values = {
        "TransactionDT": transaction_dt,
        "TransactionAmt": 42.0,
        "card1": 13926,
        "addr1": 299.0,
        "addr2": 87.0,
    }
    values.update(overrides)

    return HistoryTransaction(**values)


def _row(transaction, history) -> pd.Series:
    return build_serving_features(transaction, history).iloc[0]


# --- shape and contract -------------------------------------------------------


def test_the_frame_carries_exactly_the_locked_contract_columns():
    features = build_serving_features(_transaction(), [])

    assert list(features.columns) == list(MODEL_FEATURES)
    assert len(features) == 1


def test_the_serving_dtypes_cover_every_input_the_features_read():
    assert set(BEHAVIORAL_INPUT_COLUMNS) <= set(SERVING_DTYPES)
    assert set(CATEGORICAL_FEATURES) <= set(SERVING_DTYPES)
    assert set(CD_FEATURES) <= set(SERVING_DTYPES)


@pytest.mark.parametrize("feature", CATEGORICAL_FEATURES)
def test_a_categorical_column_is_held_as_text_not_as_object(feature):
    features = build_serving_features(_transaction(), [])

    assert str(features[feature].dtype) == "str"


@pytest.mark.parametrize("feature", CD_FEATURES)
def test_a_c_d_column_is_held_as_float64(feature):
    features = build_serving_features(_transaction(), [])

    assert features[feature].dtype == np.dtype("float64")


def test_an_absent_categorical_stays_absent_rather_than_becoming_the_text_nan():
    features = build_serving_features(_transaction(), [])

    assert pd.isna(features["R_emaildomain"].iloc[0])
    assert features["R_emaildomain"].iloc[0] != "nan"


def test_an_absent_c_d_value_stays_absent():
    features = build_serving_features(_transaction(), [])

    assert pd.isna(features["D2"].iloc[0])


def test_the_c_d_values_that_were_supplied_come_through_unchanged():
    row = _row(_transaction(), [])

    assert row["C1"] == 1.0
    assert row["C11"] == 2.0
    assert row["D15"] == 0.0


# --- one inference path -------------------------------------------------------


def test_the_features_are_the_offline_function_applied_to_the_same_rows():
    """No serving-side feature logic exists: this is ``build_behavioral_features``."""
    transaction = _transaction()
    history = [_history(CURRENT_DT - 10), _history(CURRENT_DT - 5_000)]

    combined = pd.DataFrame(
        [
            {column: getattr(entry, column) for column in BEHAVIORAL_INPUT_COLUMNS}
            for entry in [*history, transaction]
        ],
        columns=list(BEHAVIORAL_INPUT_COLUMNS),
    ).astype({column: SERVING_DTYPES[column] for column in BEHAVIORAL_INPUT_COLUMNS})

    combined = combined.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    expected = build_behavioral_features(combined).iloc[-1]
    actual = _row(transaction, history)

    for feature in (
        "card_txn_count_1h",
        "card_txn_count_24h",
        "card_amount_sum_24h",
        "address_txn_count_1h",
        "address_txn_count_24h",
        "address_amount_sum_24h",
    ):
        assert float(actual[feature]) == float(expected[feature]), feature


# --- requirement 18: strictly prior history only ------------------------------


def test_without_history_every_behavioral_count_is_zero():
    row = _row(_transaction(), [])

    assert row["card_txn_count_1h"] == 0.0
    assert row["card_txn_count_24h"] == 0.0
    assert row["address_txn_count_1h"] == 0.0
    assert row["address_txn_count_24h"] == 0.0
    assert row["card_amount_sum_24h"] == 0.0
    assert row["address_amount_sum_24h"] == 0.0


def test_without_history_both_history_flags_are_false():
    row = _row(_transaction(), [])

    assert bool(row["card_has_history_1h"]) is False
    assert bool(row["card_has_history_24h"]) is False
    assert bool(row["address_has_history_1h"]) is False


def test_a_strictly_prior_row_on_the_same_card_is_counted():
    row = _row(_transaction(), [_history(CURRENT_DT - 10)])

    assert row["card_txn_count_1h"] == 1.0
    assert row["card_txn_count_24h"] == 1.0
    assert bool(row["card_has_history_1h"]) is True
    assert bool(row["card_has_history_24h"]) is True


def test_the_amount_sum_is_the_sum_of_the_window_only():
    history = [
        _history(CURRENT_DT - 10, TransactionAmt=25.0),
        _history(CURRENT_DT - 20, TransactionAmt=17.5),
    ]

    row = _row(_transaction(), history)

    assert row["card_amount_sum_24h"] == pytest.approx(42.5)
    assert row["address_amount_sum_24h"] == pytest.approx(42.5)


def test_the_current_transaction_never_counts_itself():
    """Two prior rows, and the count is two — not three."""
    history = [_history(CURRENT_DT - 10), _history(CURRENT_DT - 20)]

    assert _row(_transaction(), history)["card_txn_count_24h"] == 2.0


@pytest.mark.parametrize(
    ("offset", "expected_1h", "expected_24h"),
    [
        (1, 1.0, 1.0),
        (ONE_HOUR - 1, 1.0, 1.0),
        (ONE_HOUR, 0.0, 1.0),
        (ONE_HOUR + 1, 0.0, 1.0),
        (ONE_DAY - 1, 0.0, 1.0),
        (ONE_DAY, 0.0, 0.0),
        (ONE_DAY + 1, 0.0, 0.0),
    ],
)
def test_the_window_edges_are_open_at_both_ends(offset, expected_1h, expected_24h):
    row = _row(_transaction(), [_history(CURRENT_DT - offset)])

    assert row["card_txn_count_1h"] == expected_1h
    assert row["card_txn_count_24h"] == expected_24h


def test_a_row_exactly_on_the_hour_boundary_leaves_the_24h_window_open():
    row = _row(_transaction(), [_history(CURRENT_DT - ONE_HOUR)])

    assert bool(row["card_has_history_1h"]) is False
    assert bool(row["card_has_history_24h"]) is True


def test_a_row_exactly_on_the_day_boundary_is_outside_both_windows():
    row = _row(_transaction(), [_history(CURRENT_DT - ONE_DAY)])

    assert bool(row["card_has_history_1h"]) is False
    assert bool(row["card_has_history_24h"]) is False


# --- entity keys --------------------------------------------------------------


def test_a_different_card_does_not_contribute_to_the_card_features():
    row = _row(_transaction(), [_history(CURRENT_DT - 10, card1=99999)])

    assert row["card_txn_count_24h"] == 0.0
    assert row["card_amount_sum_24h"] == 0.0


def test_a_different_card_still_contributes_to_the_address_features():
    row = _row(_transaction(), [_history(CURRENT_DT - 10, card1=99999)])

    assert row["address_txn_count_24h"] == 1.0


def test_a_different_address_does_not_contribute_to_the_address_features():
    row = _row(_transaction(), [_history(CURRENT_DT - 10, addr1=111.0)])

    assert row["address_txn_count_24h"] == 0.0
    assert row["address_amount_sum_24h"] == 0.0


def test_a_different_address_still_contributes_to_the_card_features():
    row = _row(_transaction(), [_history(CURRENT_DT - 10, addr1=111.0)])

    assert row["card_txn_count_24h"] == 1.0


def test_both_address_parts_must_match():
    row = _row(_transaction(), [_history(CURRENT_DT - 10, addr2=88.0)])

    assert row["address_txn_count_24h"] == 0.0


def test_a_history_row_without_an_address_matches_nothing():
    row = _row(_transaction(), [_history(CURRENT_DT - 10, addr1=None, addr2=None)])

    assert row["address_txn_count_24h"] == 0.0
    assert bool(row["address_has_history_1h"]) is False


def test_a_transaction_without_an_address_gets_zero_address_features():
    row = _row(_transaction(addr1=None, addr2=None), [_history(CURRENT_DT - 10)])

    assert row["address_txn_count_24h"] == 0.0
    assert row["address_txn_count_1h"] == 0.0


def test_a_history_row_without_a_card_matches_nothing():
    row = _row(_transaction(), [_history(CURRENT_DT - 10, card1=None)])

    assert row["card_txn_count_24h"] == 0.0
    assert bool(row["card_has_history_1h"]) is False


def test_an_integer_card_key_matches_a_float_card_key_of_the_same_value():
    """``card1`` is int64 offline and float64 here; grouping is by value."""
    row = _row(_transaction(card1=13926), [_history(CURRENT_DT - 10, card1=13926)])

    assert row["card_txn_count_24h"] == 1.0


def test_unsorted_history_is_sorted_before_the_features_are_built():
    history = [
        _history(CURRENT_DT - 5_000),
        _history(CURRENT_DT - 10),
        _history(CURRENT_DT - 90_000),
    ]

    assert _row(_transaction(), history)["card_txn_count_24h"] == 2.0


# --- temporal features of the scored row --------------------------------------


def test_the_hour_bucket_comes_from_the_scored_transaction():
    row = _row(_transaction(TransactionDT=ONE_HOUR * 5 + 60), [])

    assert row["transaction_hour_bucket"] == 5


def test_the_day_of_week_comes_from_the_scored_transaction():
    row = _row(_transaction(TransactionDT=ONE_DAY * 9), [])

    assert row["transaction_day_mod7"] == 9 % 7


def test_the_amount_and_its_log_come_from_the_scored_transaction():
    row = _row(_transaction(TransactionAmt=117.5), [_history(CURRENT_DT - 10)])

    assert row["TransactionAmt"] == 117.5
    assert row["amount_log1p"] == pytest.approx(np.log1p(117.5))


# --- the internal guard -------------------------------------------------------


def test_a_history_row_at_the_current_timestamp_is_refused_by_the_feature_layer():
    """The schema rejects this first; the guard is what stops it reaching the model."""
    transaction = _transaction()
    same_instant = _history(CURRENT_DT)

    with pytest.raises(ValueError, match="strictly less than"):
        build_serving_features(transaction, [same_instant])


def test_a_future_history_row_is_refused_by_the_feature_layer():
    transaction = _transaction()
    future = _history(CURRENT_DT + 60)

    with pytest.raises(ValueError, match="strictly less than"):
        build_serving_features(transaction, [future])


def test_the_feature_layer_guard_names_the_offending_row():
    transaction = _transaction()

    with pytest.raises(ValueError) as error:
        build_serving_features(transaction, [_history(CURRENT_DT - 5), _history(CURRENT_DT)])

    assert "history[1]" in str(error.value)
