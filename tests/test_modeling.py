import pandas as pd
import pytest

from ml.data.modeling import (
    BEHAVIORAL_FEATURES,
    CATEGORICAL_FEATURES,
    CD_FEATURES,
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


def make_transaction_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "ProductCD": ["W", "H", "C"],
            "card6": ["debit", "credit", "debit"],
            "P_emaildomain": ["gmail.com", None, "yahoo.com"],
            "R_emaildomain": [None, None, "gmail.com"],
            "C1": [1.0, 2.0, 3.0],
            "C2": [1.0, 2.0, 3.0],
            "C11": [5.0, 6.0, 7.0],
            "C13": [8.0, 9.0, 10.0],
            "D2": [None, 1.0, 2.0],
            "D3": [0.0, 1.0, 2.0],
            "D4": [None, 3.0, 4.0],
            "D8": [None, None, 5.0],
            "D10": [1.0, 2.0, 3.0],
            "D11": [0.0, 1.0, 2.0],
            "D15": [None, 4.0, 5.0],
        }
    )


def make_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
        }
    )


def build(
    behavioral=None,
    transaction_features=None,
    labels=None,
):
    return build_modeling_frame(
        make_behavioral_frame() if behavioral is None else behavioral,
        (
            make_transaction_features()
            if transaction_features is None
            else transaction_features
        ),
        make_labels() if labels is None else labels,
    )


def test_build_modeling_frame_attaches_labels():
    result = build()

    assert len(result.data) == 3
    assert result.data["isFraud"].tolist() == [0, 1, 0]


def test_model_feature_contract_excludes_identifiers():
    result = build()

    assert "TransactionID" not in result.feature_columns
    assert "TransactionDT" not in result.feature_columns
    assert "card1" not in result.feature_columns
    assert "addr1" not in result.feature_columns
    assert "addr2" not in result.feature_columns
    assert "transaction_day_index" not in result.feature_columns


def test_transaction_dt_is_retained_for_temporal_split():
    result = build()

    assert "TransactionDT" in result.data.columns
    assert result.data["TransactionDT"].tolist() == [100, 200, 300]


def test_feature_groups_are_consistent():
    assert set(CATEGORICAL_FEATURES).isdisjoint(NUMERIC_FEATURES)
    assert set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES) == set(
        MODEL_FEATURES
    )


def test_official_contract_is_28_features():
    assert len(BEHAVIORAL_FEATURES) == 13
    assert len(CATEGORICAL_FEATURES) == 4
    assert len(CD_FEATURES) == 11
    assert len(MODEL_FEATURES) == 28
    assert len(NUMERIC_FEATURES) == 24
    assert len(set(MODEL_FEATURES)) == 28


def test_d5_is_excluded_from_the_contract():
    assert "D5" not in CD_FEATURES
    assert "D5" not in MODEL_FEATURES
    assert "D5" not in NUMERIC_FEATURES


def test_numeric_features_are_behavioral_plus_cd():
    assert NUMERIC_FEATURES == [*BEHAVIORAL_FEATURES, *CD_FEATURES]


def test_all_28_features_land_in_the_frame():
    result = build()

    assert result.feature_columns == MODEL_FEATURES
    assert all(
        feature in result.data.columns for feature in MODEL_FEATURES
    )


def test_cd_missingness_is_allowed():
    result = build()

    assert result.data["D2"].isna().tolist() == [True, False, False]
    assert result.data["D8"].isna().tolist() == [True, True, False]
    assert result.data["D15"].isna().tolist() == [True, False, False]


def test_categorical_missingness_is_preserved():
    result = build()

    assert result.data["P_emaildomain"].isna().tolist() == [
        False,
        True,
        False,
    ]
    assert result.data["R_emaildomain"].isna().tolist() == [
        True,
        True,
        False,
    ]


def test_missing_behavioral_values_are_rejected():
    behavioral = make_behavioral_frame()
    behavioral.loc[1, "card_txn_count_24h"] = None

    with pytest.raises(ValueError, match="missing values"):
        build(behavioral=behavioral)


def test_missing_behavioral_column_is_rejected():
    behavioral = make_behavioral_frame().drop(
        columns=["address_amount_sum_24h"]
    )

    with pytest.raises(ValueError, match="missing required columns"):
        build(behavioral=behavioral)


def test_missing_cd_column_is_rejected():
    transaction_features = make_transaction_features().drop(columns=["C13"])

    with pytest.raises(ValueError, match="missing required columns"):
        build(transaction_features=transaction_features)


def test_duplicate_transaction_features_are_rejected():
    transaction_features = pd.DataFrame(
        {
            **{
                column: [1, 1, 3]
                for column in ["TransactionID"]
            },
            **{
                column: ["a", "b", "c"]
                for column in CATEGORICAL_FEATURES
            },
            **{
                column: [1.0, 2.0, 3.0]
                for column in CD_FEATURES
            },
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        build(transaction_features=transaction_features)


def test_duplicate_labels_are_rejected():
    labels = pd.DataFrame(
        {
            "TransactionID": [1, 1, 2],
            "isFraud": [0, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        build(labels=labels)


def test_invalid_target_is_rejected():
    labels = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="0/1"):
        build(labels=labels)


def test_missing_label_is_rejected():
    labels = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "isFraud": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="row count"):
        build(labels=labels)
