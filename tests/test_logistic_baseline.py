import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)
from ml.models.logistic_baseline import (
    build_logistic_baseline,
    fit_logistic_baseline,
    predict_fraud_probability,
)
from ml.models.preprocessing import (
    CATEGORICAL_MISSING_TOKEN,
    build_categorical_preprocessor,
)


def make_training_frame() -> pd.DataFrame:
    """A 28-feature frame with both classes and realistic missingness."""
    rows = 12

    data = {
        "TransactionAmt": [10.0 * i for i in range(1, rows + 1)],
        "amount_log1p": [float(i) / 2 for i in range(1, rows + 1)],
        "transaction_hour_bucket": [i % 24 for i in range(rows)],
        "transaction_day_mod7": [i % 7 for i in range(rows)],
        "card_txn_count_1h": [i % 4 for i in range(rows)],
        "card_txn_count_24h": [i % 6 for i in range(rows)],
        "card_has_history_1h": [bool(i % 2) for i in range(rows)],
        "card_has_history_24h": [bool(i % 3) for i in range(rows)],
        "card_amount_sum_24h": [5.0 * i for i in range(rows)],
        "address_txn_count_1h": [i % 3 for i in range(rows)],
        "address_txn_count_24h": [i % 5 for i in range(rows)],
        "address_has_history_1h": [bool(i % 2) for i in range(rows)],
        "address_amount_sum_24h": [7.0 * i for i in range(rows)],
        "ProductCD": ["W", "H", "C"] * 4,
        "card6": ["debit", "credit"] * 6,
        "P_emaildomain": ["gmail.com", "yahoo.com"] * 6,
        "R_emaildomain": [None] * 8 + ["gmail.com"] * 4,
        "isFraud": [0] * 8 + [1] * 4,
    }

    for offset, column in enumerate(CD_FEATURES):
        values = [float(i + offset) for i in range(1, rows + 1)]
        values[0] = np.nan
        data[column] = values

    return pd.DataFrame(data)


def categorical_block() -> Pipeline:
    return build_logistic_baseline().pipeline.named_steps[
        "preprocessor"
    ].transformers[1][1]


def test_logistic_preprocessor_covers_the_full_contract():
    preprocessor = build_logistic_baseline().pipeline.named_steps["preprocessor"]

    assert isinstance(preprocessor, ColumnTransformer)
    assert preprocessor.remainder == "drop"

    covered = set(preprocessor.transformers[0][2]) | set(
        preprocessor.transformers[1][2]
    )

    assert covered == set(MODEL_FEATURES)
    assert len(covered) == 28


def test_logistic_categorical_block_matches_the_shared_contract():
    """The logistic baseline must not keep a private categorical imputation rule."""

    expected = build_categorical_preprocessor()
    actual = categorical_block()

    assert list(actual.named_steps) == list(expected.named_steps)

    for step in expected.named_steps:
        assert actual.named_steps[step].get_params() == expected.named_steps[
            step
        ].get_params()


def test_logistic_categorical_missingness_uses_the_reserved_token():
    block = categorical_block()

    imputer = block.named_steps["imputer"]
    assert isinstance(imputer, SimpleImputer)
    assert imputer.strategy == "constant"
    assert imputer.fill_value == CATEGORICAL_MISSING_TOKEN

    one_hot = block.named_steps["one_hot"]
    assert isinstance(one_hot, OneHotEncoder)
    assert one_hot.handle_unknown == "ignore"
    assert one_hot.sparse_output is True


def test_logistic_no_longer_uses_most_frequent_imputation():
    params = repr(build_logistic_baseline().pipeline.get_params())

    assert "most_frequent" not in params


def test_logistic_numeric_block_keeps_median_imputer_and_scaler():
    """Only the categorical side changed; the linear model still needs scaling."""

    preprocessor = build_logistic_baseline().pipeline.named_steps["preprocessor"]
    numeric = preprocessor.transformers[0][1]

    assert list(numeric.named_steps) == ["imputer", "scaler"]
    assert list(preprocessor.transformers[0][2]) == NUMERIC_FEATURES

    imputer = numeric.named_steps["imputer"]
    assert isinstance(imputer, SimpleImputer)
    assert imputer.strategy == "median"
    assert imputer.add_indicator is False

    assert isinstance(numeric.named_steps["scaler"], StandardScaler)


def test_logistic_classifier_and_hyperparameters_are_unchanged():
    model = build_logistic_baseline()

    classifier = model.pipeline.named_steps["classifier"]

    assert isinstance(classifier, LogisticRegression)
    assert classifier.class_weight == "balanced"
    assert classifier.max_iter == 1_000
    assert classifier.solver == "lbfgs"
    assert classifier.random_state == 42

    assert model.feature_columns == MODEL_FEATURES
    assert model.categorical_features == CATEGORICAL_FEATURES
    assert model.numeric_features == NUMERIC_FEATURES


def test_fit_and_predict_returns_valid_probabilities():
    train = make_training_frame()

    model = fit_logistic_baseline(train)
    predictions = predict_fraud_probability(model, train)

    assert len(predictions) == len(train)
    assert predictions.name == "fraud_probability"
    assert predictions.notna().all()
    assert ((predictions >= 0) & (predictions <= 1)).all()


def test_fit_encodes_categorical_missingness_as_a_real_level():
    train = make_training_frame()

    model = fit_logistic_baseline(train)

    one_hot = model.pipeline.named_steps["preprocessor"].named_transformers_[
        "categorical"
    ].named_steps["one_hot"]

    r_email_levels = dict(
        zip(CATEGORICAL_FEATURES, one_hot.categories_)
    )["R_emaildomain"]

    assert CATEGORICAL_MISSING_TOKEN in set(r_email_levels)
    assert None not in set(r_email_levels)


def test_predict_tolerates_an_unseen_categorical_level():
    train = make_training_frame()

    model = fit_logistic_baseline(train)

    unseen = make_training_frame()
    unseen["ProductCD"] = "LEVEL_ABSENT_FROM_TRAIN"

    predictions = predict_fraud_probability(model, unseen)

    assert len(predictions) == len(unseen)
    assert predictions.notna().all()
