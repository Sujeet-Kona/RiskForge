import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    MODEL_FEATURES,
)
from ml.models.xgboost_model import (
    build_xgboost_model,
    fit_xgboost_model,
    predict_fraud_probability,
)


def make_training_frame() -> pd.DataFrame:
    """A 28-feature training frame that exercises both missingness families."""
    data = {
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
        "ProductCD": ["W", "H", "C", "W", "H", "C"],
        "card6": ["debit", "credit", "debit", "credit", "debit", "credit"],
        "P_emaildomain": [
            "gmail.com",
            None,
            "yahoo.com",
            "gmail.com",
            None,
            "yahoo.com",
        ],
        "R_emaildomain": [None, None, "gmail.com", None, None, "gmail.com"],
        "isFraud": [0, 0, 0, 0, 1, 1],
    }

    for index, column in enumerate(CD_FEATURES):
        values = [float(row + index) for row in range(1, 7)]

        # D8 and D15 are sparsely populated upstream, so mirror that here.
        if column in {"D8", "D15"}:
            values[0] = None
            values[3] = None

        data[column] = values

    return pd.DataFrame(data)


def test_build_xgboost_model_uses_training_class_balance():
    train = make_training_frame()

    model = build_xgboost_model(train)

    assert model.scale_pos_weight == pytest.approx(2.0)
    assert model.feature_columns == MODEL_FEATURES
    assert len(model.feature_columns) == 28


def test_build_xgboost_model_returns_an_unfitted_preprocessing_pipeline():
    model = build_xgboost_model(make_training_frame())

    assert isinstance(model.model, Pipeline)
    assert model.model.steps[0][0] == "preprocessor"
    assert model.model.steps[1][0] == "classifier"

    classifier = model.model.named_steps["classifier"]

    assert classifier.n_estimators == 300
    assert classifier.max_depth == 6
    assert classifier.learning_rate == 0.05
    assert classifier.min_child_weight == 5
    assert classifier.subsample == 0.8
    assert classifier.colsample_bytree == 0.8
    assert classifier.reg_lambda == 1.0
    assert classifier.random_state == 42
    assert classifier.tree_method == "hist"
    assert classifier.objective == "binary:logistic"
    assert classifier.eval_metric == "aucpr"
    assert classifier.scale_pos_weight == pytest.approx(2.0)


def test_build_xgboost_model_rejects_missing_features():
    train = make_training_frame().drop(columns=["TransactionAmt"])

    with pytest.raises(ValueError, match="missing required columns"):
        build_xgboost_model(train)


def test_build_xgboost_model_rejects_missing_categorical_feature():
    train = make_training_frame().drop(columns=["card6"])

    with pytest.raises(ValueError, match="missing required columns"):
        build_xgboost_model(train)


def test_build_xgboost_model_rejects_missing_cd_feature():
    train = make_training_frame().drop(columns=["C13"])

    with pytest.raises(ValueError, match="missing required columns"):
        build_xgboost_model(train)


def test_build_xgboost_model_rejects_missing_positive_class():
    train = make_training_frame()
    train["isFraud"] = 0

    with pytest.raises(ValueError, match="no positive"):
        build_xgboost_model(train)


def test_fit_accepts_string_categoricals_and_cd_missingness():
    train = make_training_frame()

    assert all(
        pd.api.types.is_string_dtype(train[column])
        for column in CATEGORICAL_FEATURES
    )
    assert train["D8"].isna().any()

    model = fit_xgboost_model(train)

    assert model.model.named_steps["preprocessor"] is not None


def test_fit_and_predict_returns_valid_scores():
    train = make_training_frame()

    model = fit_xgboost_model(train)
    predictions = predict_fraud_probability(model, train)

    assert len(predictions) == len(train)
    assert predictions.name == "fraud_probability"
    assert predictions.notna().all()
    assert ((predictions >= 0) & (predictions <= 1)).all()


def test_predict_tolerates_a_categorical_level_absent_from_training():
    train = make_training_frame()

    model = fit_xgboost_model(train)

    unseen = make_training_frame()
    unseen["ProductCD"] = "LEVEL_ABSENT_FROM_TRAIN"

    predictions = predict_fraud_probability(model, unseen)

    assert len(predictions) == len(unseen)
    assert predictions.notna().all()


def test_predict_rejects_missing_features():
    train = make_training_frame()

    model = fit_xgboost_model(train)

    incomplete = train.drop(columns=["TransactionAmt"])

    with pytest.raises(ValueError, match="missing features"):
        predict_fraud_probability(model, incomplete)
