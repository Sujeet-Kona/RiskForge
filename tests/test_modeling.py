import pandas as pd
import pytest

from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_modeling_frame,
)


def make_behavioral_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [100, 200, 300],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "amount_log1p": [2.4, 3.0, 3.4],
            "transaction_hour_bucket": [1, 2, 3],
            "transaction_day_mod7": [0, 0, 0],
            "card_txn_count_1h": [0, 1, 2],
            "card_txn_count_24h": [0, 1, 2],
            "card_has_history_1h": [False, True, True],
            "card_has_history_24h": [False, True, True],
            "card_amount_sum_24h": [0.0, 10.0, 30.0],
            "address_txn_count_1h": [0, 1, 2],
            "address_txn_count_24h": [0, 1, 2],
            "address_has_history_1h": [False, True, True],
            "address_has_history_24h": [False, True, True],
            "address_amount_sum_24h": [0.0, 10.0, 30.0],
        }
    )


def make_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
        }
    )


def test_build_modeling_frame_attaches_labels():
    result = build_modeling_frame(
        make_behavioral_frame(),
        make_labels(),
    )

    assert len(result.data) == 3
    assert result.data["isFraud"].tolist() == [0, 1, 0]


def test_model_feature_contract_excludes_identifiers():
    result = build_modeling_frame(
        make_behavioral_frame(),
        make_labels(),
    )

    assert "TransactionID" not in result.feature_columns
    assert "TransactionDT" not in result.feature_columns
    assert "card1" not in result.feature_columns
    assert "addr1" not in result.feature_columns
    assert "addr2" not in result.feature_columns
    assert "transaction_day_index" not in result.feature_columns


def test_transaction_dt_is_retained_for_temporal_split():
    result = build_modeling_frame(
        make_behavioral_frame(),
        make_labels(),
    )

    assert "TransactionDT" in result.data.columns
    assert result.data["TransactionDT"].tolist() == [100, 200, 300]


def test_feature_groups_are_consistent():
    assert set(CATEGORICAL_FEATURES).isdisjoint(NUMERIC_FEATURES)
    assert set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES) == set(
        MODEL_FEATURES
    )


def test_duplicate_labels_are_rejected():
    labels = pd.DataFrame(
        {
            "TransactionID": [1, 1, 2],
            "isFraud": [0, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_modeling_frame(
            make_behavioral_frame(),
            labels,
        )


def test_invalid_target_is_rejected():
    labels = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="0/1"):
        build_modeling_frame(
            make_behavioral_frame(),
            labels,
        )


def test_missing_label_is_rejected():
    labels = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "isFraud": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="row count"):
        build_modeling_frame(
            make_behavioral_frame(),
            labels,
        )
