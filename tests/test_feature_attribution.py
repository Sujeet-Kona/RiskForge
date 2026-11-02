from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from explain.feature_attribution import (
    CONTRIBUTION_NAME,
    FeatureAttributionMap,
    build_feature_attribution_map,
    group_contributions,
)
from ml.data.modeling import (
    BEHAVIORAL_FEATURES,
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)
from ml.models.preprocessing import (
    build_categorical_preprocessor,
    build_feature_preprocessor,
)


MISSING_INDICATOR_MARKER = "missingindicator"


def _indicator_positions(attribution: FeatureAttributionMap) -> list[int]:
    """Locate indicator columns without depending on sklearn's naming style.

    sklearn has rendered these as both ``numeric__D2_missingindicator`` and
    ``numeric__missingindicator_D2``; the marker substring is stable across
    both, and the source feature name always appears in the column.
    """
    return [
        position
        for position, name in enumerate(attribution.transformed_names)
        if MISSING_INDICATOR_MARKER in name
    ]


@pytest.fixture(scope="module")
def tiny_preprocessor(tiny_frame: pd.DataFrame) -> ColumnTransformer:
    """The real §14 preprocessor, fitted on the tiny contract frame."""
    preprocessor = build_feature_preprocessor()
    preprocessor.fit(tiny_frame[MODEL_FEATURES])

    return preprocessor


@pytest.fixture(scope="module")
def attribution(tiny_preprocessor: ColumnTransformer) -> FeatureAttributionMap:
    return build_feature_attribution_map(tiny_preprocessor)


def _refit_categorical_with(
    one_hot: OneHotEncoder,
    frame: pd.DataFrame,
) -> ColumnTransformer:
    """The real preprocessor, but with a differently configured one-hot step."""
    block = build_categorical_preprocessor()
    block.set_params(one_hot=one_hot)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                NUMERIC_FEATURES,
            ),
            ("categorical", block, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    preprocessor.fit(frame[MODEL_FEATURES])

    return preprocessor


# --- the mapping is a partition of the transformed space --------------------


def test_every_transformed_column_is_traced_to_exactly_one_contract_feature(
    attribution: FeatureAttributionMap,
):
    positions = sorted(
        position
        for feature in attribution.contract_features
        for position in attribution.columns_by_feature[feature]
    )

    assert positions == list(range(attribution.n_transformed_columns))


def test_every_contract_feature_owns_at_least_one_column(
    attribution: FeatureAttributionMap,
):
    uncovered = [
        feature
        for feature in MODEL_FEATURES
        if not attribution.columns_by_feature[feature]
    ]

    assert uncovered == []


def test_the_transformed_names_are_the_preprocessors_own(
    attribution: FeatureAttributionMap,
    tiny_preprocessor: ColumnTransformer,
):
    assert attribution.transformed_names == list(tiny_preprocessor.get_feature_names_out())


def test_source_features_parallel_the_transformed_names(
    attribution: FeatureAttributionMap,
):
    assert len(attribution.source_features) == attribution.n_transformed_columns


# --- numeric block ----------------------------------------------------------


@pytest.mark.parametrize("feature", BEHAVIORAL_FEATURES)
def test_a_behavioral_feature_without_gaps_occupies_a_single_column(
    attribution: FeatureAttributionMap,
    feature: str,
):
    assert len(attribution.columns_by_feature[feature]) == 1


def test_missing_indicators_are_attributed_to_the_feature_they_describe(
    attribution: FeatureAttributionMap,
):
    indicators = _indicator_positions(attribution)

    assert indicators, "the tiny frame injects C/D gaps, so indicators must exist"

    for position in indicators:
        source = attribution.source_features[position]
        assert source in CD_FEATURES
        assert source in attribution.transformed_names[position]


def test_the_indicator_positions_match_the_fitted_imputer(
    attribution: FeatureAttributionMap,
    tiny_preprocessor: ColumnTransformer,
):
    """The mapping is read off the fitted estimator, not parsed from names."""
    imputer = tiny_preprocessor.transformers_[0][1]
    block_columns = list(tiny_preprocessor.transformers_[0][2])

    expected = [
        block_columns[position] for position in imputer.indicator_.features_
    ]
    actual = [
        attribution.source_features[position]
        for position in _indicator_positions(attribution)
    ]

    assert actual == expected


def test_an_indicator_shares_its_feature_with_the_imputed_column(
    attribution: FeatureAttributionMap,
):
    feature = attribution.source_features[_indicator_positions(attribution)[0]]
    names = [
        attribution.transformed_names[position]
        for position in attribution.columns_by_feature[feature]
    ]

    assert len(names) == 2
    assert names[0] == f"numeric__{feature}"
    assert MISSING_INDICATOR_MARKER in names[1]


# --- categorical block ------------------------------------------------------


@pytest.mark.parametrize("feature", CATEGORICAL_FEATURES)
def test_a_categorical_feature_owns_one_column_per_fitted_level(
    attribution: FeatureAttributionMap,
    tiny_preprocessor: ColumnTransformer,
    feature: str,
):
    block = tiny_preprocessor.transformers_[1][1]
    encoder = block.named_steps["one_hot"]
    position = CATEGORICAL_FEATURES.index(feature)

    assert len(attribution.columns_by_feature[feature]) == len(encoder.categories_[position])


@pytest.mark.parametrize("feature", CATEGORICAL_FEATURES)
def test_a_categorical_feature_owns_contiguous_columns(
    attribution: FeatureAttributionMap,
    feature: str,
):
    positions = attribution.columns_by_feature[feature]

    assert positions == list(range(positions[0], positions[0] + len(positions)))


@pytest.mark.parametrize("feature", CATEGORICAL_FEATURES)
def test_every_column_of_a_categorical_feature_names_that_feature(
    attribution: FeatureAttributionMap,
    feature: str,
):
    for position in attribution.columns_by_feature[feature]:
        assert attribution.transformed_names[position].startswith(f"categorical__{feature}_")


def test_the_reserved_missing_level_is_attributed_to_its_feature(
    attribution: FeatureAttributionMap,
):
    missing_columns = [
        position
        for position, name in enumerate(attribution.transformed_names)
        if name.endswith("__MISSING__")
    ]

    assert missing_columns, "the tiny frame injects email-domain gaps"

    for position in missing_columns:
        assert attribution.source_features[position] in CATEGORICAL_FEATURES


# --- refusal to guess -------------------------------------------------------


def test_a_non_column_transformer_is_rejected():
    with pytest.raises(TypeError, match="ColumnTransformer"):
        build_feature_attribution_map("numeric")  # type: ignore[arg-type]


def test_an_unfitted_preprocessor_is_rejected():
    with pytest.raises(ValueError, match="not fitted"):
        build_feature_attribution_map(build_feature_preprocessor())


def test_a_preprocessor_without_the_categorical_block_is_rejected(
    tiny_frame: pd.DataFrame,
):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )
    preprocessor.fit(tiny_frame[MODEL_FEATURES])

    with pytest.raises(ValueError, match="no fitted 'categorical' block"):
        build_feature_attribution_map(preprocessor)


def test_an_unexpected_block_is_rejected(tiny_frame: pd.DataFrame):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                NUMERIC_FEATURES,
            ),
            ("categorical", build_categorical_preprocessor(), CATEGORICAL_FEATURES),
            ("extra", FunctionTransformer(), ["TransactionDT"]),
        ],
        remainder="drop",
    )
    frame = tiny_frame[MODEL_FEATURES].assign(TransactionDT=tiny_frame["TransactionDT"])
    preprocessor.fit(frame)

    with pytest.raises(ValueError, match="Unexpected preprocessor block 'extra'"):
        build_feature_attribution_map(preprocessor)


@pytest.mark.parametrize(
    "one_hot",
    [
        OneHotEncoder(handle_unknown="ignore", sparse_output=True, drop="first"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=2),
        OneHotEncoder(handle_unknown="ignore", sparse_output=True, max_categories=2),
    ],
    ids=["drop", "min_frequency", "max_categories"],
)
def test_a_one_hot_layout_that_breaks_the_column_arithmetic_is_rejected(
    tiny_frame: pd.DataFrame,
    one_hot: OneHotEncoder,
):
    preprocessor = _refit_categorical_with(one_hot, tiny_frame)

    with pytest.raises(ValueError, match="refuses to guess the layout"):
        build_feature_attribution_map(preprocessor)


def test_a_categorical_block_without_a_one_hot_step_is_rejected(
    tiny_frame: pd.DataFrame,
):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                        )
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    preprocessor.fit(tiny_frame[MODEL_FEATURES])

    with pytest.raises(ValueError, match="no 'one_hot' step"):
        build_feature_attribution_map(preprocessor)


# --- grouping ---------------------------------------------------------------


def test_grouping_preserves_the_total_contribution(
    attribution: FeatureAttributionMap,
):
    rng = np.random.default_rng(7)
    contributions = rng.normal(size=attribution.n_transformed_columns)

    grouped = group_contributions(attribution, contributions)

    assert grouped.sum() == pytest.approx(contributions.sum(), abs=1e-12)


def test_grouping_is_the_exact_sum_of_the_features_columns(
    attribution: FeatureAttributionMap,
):
    contributions = np.arange(attribution.n_transformed_columns, dtype=float)

    grouped = group_contributions(attribution, contributions)

    for feature in attribution.contract_features:
        assert grouped[feature] == contributions[
            attribution.columns_by_feature[feature]
        ].sum()


def test_grouping_a_flat_ones_vector_counts_columns_per_feature(
    attribution: FeatureAttributionMap,
):
    grouped = group_contributions(
        attribution, np.ones(attribution.n_transformed_columns)
    )

    for feature in attribution.contract_features:
        assert grouped[feature] == len(attribution.columns_by_feature[feature])


def test_grouping_returns_the_contract_features_in_order(
    attribution: FeatureAttributionMap,
):
    grouped = group_contributions(attribution, np.zeros(attribution.n_transformed_columns))

    assert list(grouped.index) == MODEL_FEATURES
    assert grouped.index.name == "feature"
    assert grouped.name == CONTRIBUTION_NAME


def test_grouping_rejects_the_wrong_width(attribution: FeatureAttributionMap):
    with pytest.raises(ValueError, match="transformed-column contributions"):
        group_contributions(attribution, np.zeros(attribution.n_transformed_columns + 1))


def test_grouping_rejects_a_multi_row_matrix(attribution: FeatureAttributionMap):
    wide = np.zeros((2, attribution.n_transformed_columns))

    with pytest.raises(ValueError, match="transformed-column contributions"):
        group_contributions(attribution, wide)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_grouping_rejects_non_finite_contributions(
    attribution: FeatureAttributionMap,
    bad: float,
):
    contributions = np.zeros(attribution.n_transformed_columns)
    contributions[0] = bad

    with pytest.raises(ValueError, match="non-finite"):
        group_contributions(attribution, contributions)


def test_the_map_is_frozen(attribution: FeatureAttributionMap):
    with pytest.raises(FrozenInstanceError):
        attribution.source_features = []  # type: ignore[misc]
