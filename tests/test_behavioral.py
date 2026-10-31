import pandas as pd
import pytest

from ml.features.behavioral import build_behavioral_features


def make_df(
    transaction_dt,
    amounts=None,
    cards=None,
    addr1=None,
    addr2=None,
):
    n = len(transaction_dt)

    return pd.DataFrame(
        {
            "TransactionID": range(1, n + 1),
            "TransactionDT": transaction_dt,
            "TransactionAmt": amounts or [10.0] * n,
            "card1": cards or [111] * n,
            "addr1": addr1 or [10] * n,
            "addr2": addr2 or [1] * n,
        }
    )


def test_same_timestamp_transactions_cannot_see_each_other():
    df = make_df(
        [1000, 1000, 1001],
        amounts=[10.0, 20.0, 30.0],
    )

    result = build_behavioral_features(df)

    assert result.loc[0, "card_txn_count_1h"] == 0
    assert result.loc[1, "card_txn_count_1h"] == 0

    assert result.loc[2, "card_txn_count_1h"] == 2
    assert result.loc[2, "card_amount_sum_24h"] == 30.0


def test_only_strictly_prior_transactions_are_used():
    df = make_df(
        [1000, 1000],
        amounts=[10.0, 20.0],
    )

    result = build_behavioral_features(df)

    assert result.loc[0, "card_txn_count_24h"] == 0
    assert result.loc[1, "card_txn_count_24h"] == 0
    assert result.loc[0, "card_amount_sum_24h"] == 0.0
    assert result.loc[1, "card_amount_sum_24h"] == 0.0


def test_one_hour_boundary_is_exclusive():
    df = make_df(
        [1000, 4600, 4601],
        amounts=[10.0, 20.0, 30.0],
    )

    result = build_behavioral_features(df)

    # 4600 - 1000 = 3600 exactly.
    # The 1000-second event must be excluded.
    assert result.loc[1, "card_txn_count_1h"] == 0

    # At 4601, the 1000-second event is 3601 seconds old,
    # while the 4600-second event is only 1 second old.
    assert result.loc[2, "card_txn_count_1h"] == 1


def test_24_hour_boundary_is_exclusive():
    df = make_df(
        [1000, 87400, 87401],
        amounts=[10.0, 20.0, 30.0],
    )

    result = build_behavioral_features(df)

    # 87400 - 1000 = 86400 exactly.
    assert result.loc[1, "card_txn_count_24h"] == 0

    # At 87401, the 87400 event is inside the window.
    assert result.loc[2, "card_txn_count_24h"] == 1
    assert result.loc[2, "card_amount_sum_24h"] == 20.0


def test_different_cards_have_independent_history():
    df = make_df(
        [1000, 1001, 1002],
        amounts=[10.0, 20.0, 30.0],
        cards=[111, 222, 111],
    )

    result = build_behavioral_features(df)

    assert result.loc[0, "card_txn_count_24h"] == 0
    assert result.loc[1, "card_txn_count_24h"] == 0
    assert result.loc[2, "card_txn_count_24h"] == 1
    assert result.loc[2, "card_amount_sum_24h"] == 10.0


def test_missing_behavioral_key_does_not_create_history():
    df = make_df(
        [1000, 1001, 1002],
        amounts=[10.0, 20.0, 30.0],
        cards=[111, None, 111],
    )

    result = build_behavioral_features(df)

    assert result.loc[1, "card_txn_count_24h"] == 0
    assert result.loc[2, "card_txn_count_24h"] == 1
    assert result.loc[2, "card_amount_sum_24h"] == 10.0


def test_unsorted_input_is_rejected():
    df = make_df(
        [1002, 1000, 1001],
    )

    with pytest.raises(ValueError, match="sorted by TransactionDT"):
        build_behavioral_features(df)


def test_temporal_features_have_explicit_relative_semantics():
    df = make_df(
        [0, 3600, 86400],
    )

    result = build_behavioral_features(df)

    assert "transaction_hour_bucket" in result.columns
    assert "transaction_day_index" in result.columns
    assert "transaction_day_mod7" in result.columns

    assert "transaction_day_of_week" not in result.columns


def test_amount_log_feature_is_correct():
    df = make_df(
        [1000],
        amounts=[9.0],
    )

    result = build_behavioral_features(df)

    assert result.loc[0, "amount_log1p"] == pytest.approx(2.30258509299)


def test_history_flags_match_history_counts():
    df = make_df(
        [1000, 1001, 2000],
        amounts=[10.0, 20.0, 30.0],
        cards=[111, 111, 111],
    )

    result = build_behavioral_features(df)

    assert result.loc[0, "card_txn_count_1h"] == 0
    assert bool(result.loc[0, "card_has_history_1h"]) is False

    assert result.loc[1, "card_txn_count_1h"] == 1
    assert bool(result.loc[1, "card_has_history_1h"]) is True

    assert result.loc[2, "card_txn_count_1h"] == 2
    assert bool(result.loc[2, "card_has_history_1h"]) is True

    assert result.loc[0, "card_txn_count_24h"] == 0
    assert bool(result.loc[0, "card_has_history_24h"]) is False

    assert result.loc[1, "card_txn_count_24h"] == 1
    assert bool(result.loc[1, "card_has_history_24h"]) is True

    assert result.loc[2, "card_txn_count_24h"] == 2
    assert bool(result.loc[2, "card_has_history_24h"]) is True


def test_same_timestamp_history_flags_are_false():
    df = make_df(
        [1000, 1000, 1001],
        amounts=[10.0, 20.0, 30.0],
    )

    result = build_behavioral_features(df)

    assert bool(result.loc[0, "card_has_history_1h"]) is False
    assert bool(result.loc[1, "card_has_history_1h"]) is False

    assert bool(result.loc[0, "card_has_history_24h"]) is False
    assert bool(result.loc[1, "card_has_history_24h"]) is False

    assert bool(result.loc[2, "card_has_history_1h"]) is True
    assert bool(result.loc[2, "card_has_history_24h"]) is True


def test_history_flags_respect_one_hour_boundary():
    df = make_df(
        [1000, 4600, 4601],
        amounts=[10.0, 20.0, 30.0],
    )

    result = build_behavioral_features(df)

    assert bool(result.loc[1, "card_has_history_1h"]) is False

    assert bool(result.loc[2, "card_has_history_1h"]) is True


def test_history_flags_respect_24_hour_boundary():
    df = make_df(
        [1000, 87400, 87401],
        amounts=[10.0, 20.0, 30.0],
    )

    result = build_behavioral_features(df)

    assert bool(result.loc[1, "card_has_history_24h"]) is False
    assert bool(result.loc[2, "card_has_history_24h"]) is True


def test_missing_behavioral_key_has_no_history_flag():
    df = make_df(
        [1000, 1001, 1002],
        amounts=[10.0, 20.0, 30.0],
        cards=[111, None, 111],
    )

    result = build_behavioral_features(df)

    assert bool(result.loc[1, "card_has_history_24h"]) is False

    assert result.loc[2, "card_txn_count_24h"] == 1
    assert bool(result.loc[2, "card_has_history_24h"]) is True


def test_address_history_flags_are_independent():
    df = make_df(
        [1000, 1001, 1002],
        amounts=[10.0, 20.0, 30.0],
        cards=[111, 222, 333],
        addr1=[10, 20, 10],
        addr2=[1, 2, 1],
    )

    result = build_behavioral_features(df)

    assert bool(result.loc[0, "address_has_history_24h"]) is False
    assert bool(result.loc[1, "address_has_history_24h"]) is False

    assert result.loc[2, "address_txn_count_24h"] == 1
    assert bool(result.loc[2, "address_has_history_24h"]) is True


def test_history_flag_columns_are_boolean():
    df = make_df([1000, 1001])

    result = build_behavioral_features(df)

    assert result["card_has_history_1h"].dtype == bool
    assert result["card_has_history_24h"].dtype == bool
    assert result["address_has_history_1h"].dtype == bool
    assert result["address_has_history_24h"].dtype == bool
