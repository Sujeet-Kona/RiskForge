import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)
from ml.models.preprocessing import (
    CATEGORICAL_MISSING_TOKEN,
    build_categorical_preprocessor,
    build_feature_preprocessor,
)


def to_dense(encoded):
    """Densify a block; ColumnTransformer densifies unless every block is sparse."""

    if sparse.issparse(encoded):
        return encoded.toarray()

    return np.asarray(encoded)


def make_frame() -> pd.DataFrame:
    """A 28-feature frame with no numeric gaps and one categorical gap."""
    rows = 4

    data = {
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
        "amount_log1p": [2.4, 3.0, 3.4, 3.7],
        "transaction_hour_bucket": [1, 2, 3, 4],
        "transaction_day_mod7": [0, 1, 2, 3],
        "card_txn_count_1h": [0, 1, 2, 3],
        "card_txn_count_24h": [0, 2, 4, 6],
        "card_has_history_1h": [False, True, True, True],
        "card_has_history_24h": [False, True, True, True],
        "card_amount_sum_24h": [0.0, 10.0, 30.0, 60.0],
        "address_txn_count_1h": [0, 1, 2, 3],
        "address_txn_count_24h": [0, 2, 4, 6],
        "address_has_history_1h": [False, True, True, True],
        "address_has_history_24h": [False, True, True, True],
        "address_amount_sum_24h": [0.0, 10.0, 30.0, 60.0],
        "ProductCD": ["W", "H", "C", "W"],
        "card6": ["debit", "credit", "debit", "credit"],
        "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com", "gmail.com"],
        "R_emaildomain": [None, None, "gmail.com", None],
    }

    for index, column in enumerate(CD_FEATURES):
        data[column] = [
            float(value + index) for value in range(1, rows + 1)
        ]

    return pd.DataFrame(data)


def test_preprocessor_covers_the_full_contract_exactly():
    preprocessor = build_feature_preprocessor()

    numeric_block = preprocessor.transformers[0]
    categorical_block = preprocessor.transformers[1]

    assert numeric_block[0] == "numeric"
    assert list(numeric_block[2]) == NUMERIC_FEATURES

    assert categorical_block[0] == "categorical"
    assert list(categorical_block[2]) == CATEGORICAL_FEATURES

    covered = set(numeric_block[2]) | set(categorical_block[2])

    assert covered == set(MODEL_FEATURES)
    assert len(covered) == 28


def test_numeric_block_is_median_imputer_with_missing_indicator():
    preprocessor = build_feature_preprocessor()
    numeric = preprocessor.transformers[0][1]

    assert isinstance(numeric, SimpleImputer)
    assert numeric.strategy == "median"
    assert numeric.add_indicator is True


def test_categorical_block_is_constant_imputer_then_one_hot():
    preprocessor = build_feature_preprocessor()
    categorical = preprocessor.transformers[1][1]

    assert isinstance(categorical, Pipeline)
    assert list(categorical.named_steps) == [
        "normalize_missing",
        "imputer",
        "one_hot",
    ]

    imputer = categorical.named_steps["imputer"]
    assert isinstance(imputer, SimpleImputer)
    assert imputer.strategy == "constant"
    assert imputer.fill_value == CATEGORICAL_MISSING_TOKEN

    one_hot = categorical.named_steps["one_hot"]
    assert isinstance(one_hot, OneHotEncoder)
    assert one_hot.handle_unknown == "ignore"
    assert one_hot.sparse_output is True


def test_feature_preprocessor_uses_the_shared_categorical_factory():
    """One definition of categorical missingness, reused by every model."""

    expected = build_categorical_preprocessor()
    actual = build_feature_preprocessor().transformers[1][1]

    assert list(actual.named_steps) == list(expected.named_steps)

    for step in expected.named_steps:
        assert actual.named_steps[step].get_params() == expected.named_steps[
            step
        ].get_params()


def test_preprocessor_is_unfitted_until_fit_on_train():
    preprocessor = build_feature_preprocessor()

    with pytest.raises(NotFittedError):
        preprocessor.transform(make_frame())


def test_categorical_missingness_becomes_an_encoded_level():
    preprocessor = build_feature_preprocessor().fit(make_frame())

    names = list(preprocessor.get_feature_names_out())

    assert any(
        name.endswith(f"R_emaildomain_{CATEGORICAL_MISSING_TOKEN}")
        for name in names
    )


def test_python_none_and_nan_share_one_missing_level():
    """``None`` and ``np.nan`` must collapse to one level, never a literal ``None`` category."""

    with_none = make_frame()
    with_nan = make_frame()
    with_nan["R_emaildomain"] = [np.nan, np.nan, "gmail.com", np.nan]

    names_from_none = set(
        build_feature_preprocessor().fit(with_none).get_feature_names_out()
    )
    names_from_nan = set(
        build_feature_preprocessor().fit(with_nan).get_feature_names_out()
    )

    assert names_from_none == names_from_nan
    assert not any(name.endswith("_None") for name in names_from_none)


def test_no_numeric_indicator_when_nothing_is_missing():
    preprocessor = build_feature_preprocessor().fit(make_frame())
    numeric = preprocessor.named_transformers_["numeric"]

    assert numeric.indicator_.features_.tolist() == []


def test_missing_cd_value_gets_a_median_fill_and_an_indicator():
    frame = make_frame()
    frame.loc[0, "D15"] = None

    preprocessor = build_feature_preprocessor().fit(frame)
    numeric = preprocessor.named_transformers_["numeric"]

    assert NUMERIC_FEATURES.index("D15") in numeric.indicator_.features_

    encoded = to_dense(preprocessor.transform(frame))

    assert not np.isnan(encoded).any()


def test_unknown_categorical_levels_are_ignored_not_fatal():
    preprocessor = build_feature_preprocessor().fit(make_frame())

    unseen = make_frame()
    unseen["ProductCD"] = "LEVEL_ABSENT_FROM_TRAIN"

    encoded = preprocessor.transform(unseen)

    assert encoded.shape == preprocessor.transform(make_frame()).shape


def test_identifier_and_target_columns_are_dropped():
    wide = make_frame()
    wide["TransactionID"] = [1, 2, 3, 4]
    wide["TransactionDT"] = [100, 200, 300, 400]
    wide["isFraud"] = [0, 0, 1, 1]

    preprocessor = build_feature_preprocessor().fit(wide)

    names = " ".join(preprocessor.get_feature_names_out())

    assert "TransactionID" not in names
    assert "TransactionDT" not in names
    assert "isFraud" not in names


def test_fitted_on_train_only_the_test_vocabulary_cannot_leak_in():
    train = make_frame()

    preprocessor = build_feature_preprocessor().fit(train)

    train_levels = set(
        preprocessor.named_transformers_["categorical"]
        .named_steps["one_hot"]
        .categories_[0]
    )

    assert train_levels == {"W", "H", "C"}
