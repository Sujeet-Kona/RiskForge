import pandas as pd
import pytest

from ml.models.xgboost_model import (
    build_xgboost_model,
    fit_xgboost_model,
    predict_fraud_probability,
)


def make_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "amount_log1p": [2.4, 3.0, 3.4, 3.7, 4.0, 4.1],
            "transaction_hour_bucket": [1, 2, 3, 4, 5, 6],
            "transaction_day_mod7": [0, 1, 2, 3, 4, 5],
            "card_txn_count_1h": [0, 1, 2, 0, 3, 1],
            "card_txn_count_24h": [0, 2, 4, 1, 5, 2],
            "card_has_history_1h": [False, True, True, False, True, True],
            "card_has_history_24h": [False, True, True, True, True, True],
            "card_amount_sum_24h": [0.0, 20.0, 50.0, 10.0, 100.0, 30.0],
            "address_txn_count_1h": [0, 1, 2, 1, 3, 2],
            "address_txn_count_24h": [0, 2, 4, 2, 5, 3],
            "address_has_history_1h": [False, True, True, True, True, True],
            "address_has_history_24h": [False, True, True, True, True, True],
            "address_amount_sum_24h": [0.0, 20.0, 50.0, 40.0, 100.0, 70.0],
            "isFraud": [0, 0, 0, 0, 1, 1],
        }
    )


def test_build_xgboost_model_uses_training_class_balance():
    train = make_training_frame()

    model = build_xgboost_model(train)

    assert model.scale_pos_weight == pytest.approx(2.0)
    assert model.feature_columns
    assert model.model.n_estimators == 300


def test_build_xgboost_model_rejects_missing_features():
    train = make_training_frame().drop(columns=["TransactionAmt"])

    with pytest.raises(ValueError, match="missing required columns"):
        build_xgboost_model(train)


def test_build_xgboost_model_rejects_missing_positive_class():
    train = make_training_frame()
    train["isFraud"] = 0

    with pytest.raises(ValueError, match="no positive"):
        build_xgboost_model(train)


def test_fit_and_predict_returns_valid_scores():
    train = make_training_frame()

    model = fit_xgboost_model(train)
    predictions = predict_fraud_probability(model, train)

    assert len(predictions) == len(train)
    assert predictions.name == "fraud_probability"
    assert predictions.notna().all()
    assert ((predictions >= 0) & (predictions <= 1)).all()


def test_predict_rejects_missing_features():
    train = make_training_frame()

    model = fit_xgboost_model(train)

    incomplete = train.drop(columns=["TransactionAmt"])

    with pytest.raises(ValueError, match="missing features"):
        predict_fraud_probability(model, incomplete)
