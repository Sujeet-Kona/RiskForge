import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix

from explain.feature_attribution import build_feature_attribution_map
from explain.model_explanation import (
    ATTRIBUTION_SPACE,
    DEFAULT_TOP_K,
    FLOAT32_ADDITIVITY_TOLERANCE,
    NEUTRAL,
    RISK_DECREASING,
    RISK_INCREASING,
    FeatureContribution,
    ModelExplanation,
    build_model_explainer,
)
from explain.reason_codes import reason_code_for_feature
from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
)
from ml.models.preprocessing import build_feature_preprocessor


MISSING_INDICATOR_MARKER = "missingindicator"
RESERVED_CATEGORICAL_LEVEL = "__MISSING__"

CONTRIBUTION_FIELDS = {
    "feature",
    "contribution",
    "direction",
    "reason_code",
    "reason_title",
    "n_transformed_columns",
}


class _EstimatorWithoutProbabilities(BaseEstimator):
    """A final pipeline step that can be fitted but never emits probabilities."""

    def fit(self, X, y):
        return self


MISSING_INDICATOR_MARKER = "missingindicator"
RESERVED_CATEGORICAL_LEVEL = "__MISSING__"

CONTRIBUTION_FIELDS = {
    "feature",
    "contribution",
    "direction",
    "reason_code",
    "reason_title",
    "n_transformed_columns",
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def densify(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray())

    return np.asarray(matrix)


def row(frame: pd.DataFrame, index: int) -> pd.DataFrame:
    return frame.iloc[[index]]


@pytest.fixture(scope="module")
def first_row(tiny_frame: pd.DataFrame) -> pd.DataFrame:
    return row(tiny_frame, 0)


@pytest.fixture(scope="module")
def explanation(tiny_explainer, first_row: pd.DataFrame) -> ModelExplanation:
    return tiny_explainer.explain(first_row)


@pytest.fixture(scope="module")
def dominant_negative_row(tiny_explainer, tiny_frame: pd.DataFrame) -> int:
    """The first row whose strongest contributor pushes risk *down*.

    Requirement 7 is only a real test if such a row exists, so it is located in
    the fixture rather than assumed.
    """
    for index in range(len(tiny_frame)):
        explained = tiny_explainer.explain(row(tiny_frame, index), top_k=1)
        largest = explained.contributions[0]

        if largest.direction == RISK_DECREASING:
            return index

    raise AssertionError("no tiny-frame row has a dominant risk-decreasing feature")


# --- requirement 1: determinism ---------------------------------------------


def test_the_same_row_produces_the_same_explanation(
    tiny_explainer, first_row: pd.DataFrame
):
    first = tiny_explainer.explain(first_row).to_dict()
    second = tiny_explainer.explain(first_row).to_dict()

    assert first == second


def test_a_freshly_built_explainer_reproduces_the_explanation(
    tiny_pipeline: Pipeline, first_row: pd.DataFrame
):
    rebuilt = build_model_explainer(tiny_pipeline).explain(first_row)
    original = build_model_explainer(tiny_pipeline).explain(first_row)

    assert rebuilt.to_dict() == original.to_dict()


def test_explaining_a_row_twice_does_not_mutate_the_row(
    tiny_explainer, first_row: pd.DataFrame
):
    before = first_row.copy(deep=True)

    tiny_explainer.explain(first_row)
    tiny_explainer.explain(first_row)

    pd.testing.assert_frame_equal(first_row, before)


# --- requirement 2: top_k ----------------------------------------------------


@pytest.mark.parametrize("top_k", [1, 2, 3, 5, 10, 28, 100])
def test_top_k_is_honoured(tiny_explainer, first_row: pd.DataFrame, top_k: int):
    explained = tiny_explainer.explain(first_row, top_k=top_k)

    assert explained.top_k == top_k
    assert len(explained.risk_increasing) <= top_k
    assert len(explained.risk_decreasing) <= top_k


def test_top_k_one_returns_only_the_strongest_upward_driver(
    tiny_explainer, first_row: pd.DataFrame
):
    one = tiny_explainer.explain(first_row, top_k=1)
    three = tiny_explainer.explain(first_row, top_k=3)

    assert len(one.risk_increasing) == 1
    assert one.risk_increasing[0] == three.risk_increasing[0]


def test_top_k_defaults_to_three(
    tiny_explainer, first_row: pd.DataFrame
):
    assert DEFAULT_TOP_K == 3
    assert tiny_explainer.explain(first_row).top_k == DEFAULT_TOP_K


def test_a_larger_top_k_is_a_prefix_extension(
    tiny_explainer, first_row: pd.DataFrame
):
    small = tiny_explainer.explain(first_row, top_k=2)
    large = tiny_explainer.explain(first_row, top_k=6)

    assert large.risk_increasing[:2] == small.risk_increasing
    assert large.risk_decreasing[:2] == small.risk_decreasing


# --- requirement 3: invalid top_k --------------------------------------------


@pytest.mark.parametrize("top_k", [0, -1, -100])
def test_a_non_positive_top_k_is_rejected(
    tiny_explainer, first_row: pd.DataFrame, top_k: int
):
    with pytest.raises(ValueError, match="at least 1"):
        tiny_explainer.explain(first_row, top_k=top_k)


@pytest.mark.parametrize("top_k", [1.0, 3.5, "3", None, True, [3]])
def test_a_non_integer_top_k_is_rejected(
    tiny_explainer, first_row: pd.DataFrame, top_k
):
    with pytest.raises(TypeError, match="integer"):
        tiny_explainer.explain(first_row, top_k=top_k)


# --- requirement 5: required fields ------------------------------------------


def test_every_returned_contribution_carries_its_required_fields(explanation):
    entries = [*explanation.risk_increasing, *explanation.risk_decreasing]

    assert entries

    for entry in entries:
        assert isinstance(entry, FeatureContribution)
        assert entry.feature in MODEL_FEATURES
        assert isinstance(entry.contribution, float)
        assert entry.direction in {RISK_INCREASING, RISK_DECREASING, NEUTRAL}
        assert entry.reason_code == reason_code_for_feature(entry.feature).code
        assert entry.reason_title == reason_code_for_feature(entry.feature).title
        assert entry.n_transformed_columns >= 1


def test_the_serialised_form_exposes_the_same_fields(explanation):
    payload = explanation.to_dict()

    for bucket in ("risk_increasing", "risk_decreasing"):
        for entry in payload[bucket]:
            assert set(entry) == CONTRIBUTION_FIELDS


def test_the_explanation_reports_the_attribution_space(explanation):
    assert explanation.attribution_space == ATTRIBUTION_SPACE
    assert ATTRIBUTION_SPACE == "raw_margin_log_odds"


def test_all_twenty_eight_contract_features_are_explained(explanation):
    assert {entry.feature for entry in explanation.contributions} == set(MODEL_FEATURES)


def test_contributions_are_sorted_by_magnitude(explanation):
    magnitudes = [abs(entry.contribution) for entry in explanation.contributions]

    assert magnitudes == sorted(magnitudes, reverse=True)


# --- requirement 6: positive contributions ------------------------------------


def test_every_risk_increasing_entry_is_genuinely_positive(explanation):
    assert explanation.risk_increasing

    for entry in explanation.risk_increasing:
        assert entry.contribution > 0.0
        assert entry.direction == RISK_INCREASING


def test_direction_agrees_with_the_sign_of_every_contribution(explanation):
    for entry in explanation.contributions:
        if entry.contribution > 0.0:
            assert entry.direction == RISK_INCREASING
        elif entry.contribution < 0.0:
            assert entry.direction == RISK_DECREASING
        else:
            assert entry.direction == NEUTRAL


def test_risk_increasing_is_ranked_best_first(explanation):
    values = [entry.contribution for entry in explanation.risk_increasing]

    assert values == sorted(values, reverse=True)


def test_the_totals_sum_the_whole_direction_not_just_the_top_k(explanation):
    positives = sum(
        entry.contribution
        for entry in explanation.contributions
        if entry.contribution > 0.0
    )
    negatives = sum(
        entry.contribution
        for entry in explanation.contributions
        if entry.contribution < 0.0
    )

    assert explanation.total_risk_increasing == pytest.approx(positives)
    assert explanation.total_risk_decreasing == pytest.approx(negatives)
    assert explanation.total_risk_increasing > 0.0
    assert explanation.total_risk_decreasing < 0.0


# --- requirement 7: negatives are not fraud evidence ---------------------------


def test_a_downward_driver_never_appears_as_a_risk_reason(
    tiny_explainer, tiny_frame: pd.DataFrame, dominant_negative_row: int
):
    explained = tiny_explainer.explain(
        row(tiny_frame, dominant_negative_row), top_k=3
    )
    strongest = explained.contributions[0]

    assert strongest.contribution < 0.0
    assert strongest.direction == RISK_DECREASING
    assert strongest not in explained.risk_increasing
    assert strongest.feature not in {e.feature for e in explained.risk_increasing}


def test_the_strongest_downward_driver_leads_the_decreasing_list(
    tiny_explainer, tiny_frame: pd.DataFrame, dominant_negative_row: int
):
    explained = tiny_explainer.explain(
        row(tiny_frame, dominant_negative_row), top_k=3
    )

    assert explained.risk_decreasing[0] == explained.contributions[0]


def test_top_reason_codes_only_cite_upward_drivers(
    tiny_explainer, tiny_frame: pd.DataFrame, dominant_negative_row: int
):
    explained = tiny_explainer.explain(
        row(tiny_frame, dominant_negative_row), top_k=3
    )
    decreasing = {e.reason_code for e in explained.risk_decreasing}
    increasing = {e.reason_code for e in explained.risk_increasing}

    assert set(explained.top_reason_codes) == increasing
    assert not (increasing & decreasing)


def test_risk_decreasing_is_ranked_most_negative_first(explanation):
    values = [entry.contribution for entry in explanation.risk_decreasing]

    assert values == sorted(values)


# --- requirement 19: the actual fitted model, not a surrogate ------------------


def test_the_explainer_wraps_the_pipeline_it_was_given(tiny_explainer, tiny_pipeline):
    assert tiny_explainer.pipeline is tiny_pipeline
    assert tiny_explainer.classifier is tiny_pipeline.named_steps["classifier"]
    assert tiny_explainer.preprocessor is tiny_pipeline.named_steps["preprocessor"]


def test_the_attribution_map_covers_exactly_the_models_input_width(tiny_explainer):
    classifier = tiny_explainer.classifier

    assert tiny_explainer.attribution.n_transformed_columns == classifier.n_features_in_


def test_the_base_value_is_the_boosters_own_bias(tiny_explainer, first_row):
    features = first_row[MODEL_FEATURES]
    transformed = tiny_explainer.preprocessor.transform(features)
    native = tiny_explainer.classifier.get_booster().predict(
        DMatrix(transformed), pred_contribs=True
    )

    assert tiny_explainer.explainer.expected_value == pytest.approx(
        float(native[0, -1]), abs=0.0
    )


def test_the_contributions_match_xgboosts_native_treeshap(tiny_explainer, first_row):
    """Requirement 19 in its strongest form: same numbers the booster computes."""
    features = first_row[MODEL_FEATURES]
    transformed = tiny_explainer.preprocessor.transform(features)

    native = tiny_explainer.classifier.get_booster().predict(
        DMatrix(transformed), pred_contribs=True
    )[0, :-1]
    shap_values = np.asarray(
        tiny_explainer.explainer.shap_values(transformed), dtype=float
    ).ravel()

    assert np.allclose(native, shap_values, rtol=0.0, atol=0.0)


def test_the_decomposition_closes_against_the_raw_margin(explanation):
    total = explanation.base_value + sum(
        entry.contribution for entry in explanation.contributions
    )

    assert total == pytest.approx(
        explanation.raw_margin, abs=FLOAT32_ADDITIVITY_TOLERANCE
    )


def test_the_raw_margin_reproduces_the_models_own_probability(
    tiny_explainer, first_row: pd.DataFrame
):
    explained = tiny_explainer.explain(first_row)
    direct = float(tiny_explainer.pipeline.predict_proba(first_row[MODEL_FEATURES])[0, 1])

    assert explained.raw_model_probability == direct
    assert sigmoid(explained.raw_margin) == pytest.approx(direct, abs=1e-6)


def test_no_second_model_is_fitted(tiny_explainer, first_row: pd.DataFrame):
    """The explainer is read-only: it must not disturb the locked artifact."""
    booster = tiny_explainer.classifier.get_booster()
    before = booster.save_raw("json")

    tiny_explainer.explain(first_row)

    assert booster.save_raw("json") == before


# --- requirements 17 and 18: missing values ------------------------------------


def _column_layout(tiny_explainer, feature: str) -> tuple[list[int], list[str]]:
    positions = tiny_explainer.attribution.columns_by_feature[feature]
    names = tiny_explainer.attribution.transformed_names

    return positions, [names[position] for position in positions]


def _transformed_row(tiny_explainer, transaction: pd.DataFrame) -> np.ndarray:
    return densify(
        tiny_explainer.preprocessor.transform(transaction[MODEL_FEATURES])
    )[0]


def _one_hot_row(tiny_explainer, transaction: pd.DataFrame, feature: str) -> np.ndarray:
    positions, _ = _column_layout(tiny_explainer, feature)

    return _transformed_row(tiny_explainer, transaction)[positions]


def _observed_levels(tiny_explainer) -> dict[str, str]:
    """One level per categorical feature that the encoder actually fitted on."""
    encoder = tiny_explainer.preprocessor.transformers_[1][1].named_steps["one_hot"]
    levels: dict[str, str] = {}

    for feature, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        observed = [
            str(category)
            for category in categories
            if str(category) != RESERVED_CATEGORICAL_LEVEL
        ]
        levels[feature] = observed[0]

    return levels


@pytest.fixture(scope="module")
def complete_row(tiny_explainer, first_row: pd.DataFrame) -> pd.DataFrame:
    """A gap-free transaction, so every missing indicator must be off."""
    filled = first_row.copy()
    levels = _observed_levels(tiny_explainer)

    for feature in CD_FEATURES:
        filled[feature] = 1.0

    for feature, level in levels.items():
        filled[feature] = level

    return filled


def test_a_gap_free_row_switches_every_missing_indicator_off(
    tiny_explainer, complete_row: pd.DataFrame
):
    transformed = _transformed_row(tiny_explainer, complete_row)

    checked = 0

    for feature in CD_FEATURES:
        positions, column_names = _column_layout(tiny_explainer, feature)

        for index, position in enumerate(positions):
            if MISSING_INDICATOR_MARKER in column_names[index]:
                assert transformed[position] == 0.0
                checked += 1
            else:
                assert transformed[position] == pytest.approx(1.0)

    assert checked == len(CD_FEATURES)


def test_a_missing_numeric_value_uses_the_fitted_median_and_indicator(
    tiny_explainer, complete_row: pd.DataFrame
):
    """Requirement 18: imputation comes from the fitted preprocessor, not from here."""
    feature = CD_FEATURES[0]
    gapped = complete_row.copy()
    gapped[feature] = np.nan

    positions, column_names = _column_layout(tiny_explainer, feature)
    transformed = _transformed_row(tiny_explainer, gapped)

    assert MISSING_INDICATOR_MARKER in column_names[1]
    assert transformed[positions[1]] == 1.0

    block_columns = list(tiny_explainer.preprocessor.transformers_[0][2])
    median = tiny_explainer.preprocessor.transformers_[0][1].statistics_[
        block_columns.index(feature)
    ]

    assert transformed[positions[0]] == pytest.approx(float(median))


def test_a_gap_free_categorical_row_lights_exactly_one_level(
    tiny_explainer, complete_row: pd.DataFrame
):
    for feature in CATEGORICAL_FEATURES:
        positions, column_names = _column_layout(tiny_explainer, feature)
        levels = _one_hot_row(tiny_explainer, complete_row, feature)

        assert levels.sum() == pytest.approx(1.0)

        reserved = [
            index
            for index, name in enumerate(column_names)
            if name.endswith(RESERVED_CATEGORICAL_LEVEL)
        ]

        for index in reserved:
            assert levels[index] == 0.0


def test_a_missing_categorical_value_becomes_the_reserved_level(
    tiny_explainer, complete_row: pd.DataFrame
):
    """Requirement 17: absence is encoded, exactly as ``build_categorical_preprocessor`` does."""
    seen_reserved = 0

    for feature in CATEGORICAL_FEATURES:
        positions, column_names = _column_layout(tiny_explainer, feature)
        reserved = [
            index
            for index, name in enumerate(column_names)
            if name.endswith(RESERVED_CATEGORICAL_LEVEL)
        ]

        if not reserved:
            continue

        gapped = complete_row.copy()
        gapped[feature] = None
        levels = _one_hot_row(tiny_explainer, gapped, feature)

        assert levels[reserved[0]] == 1.0
        assert levels.sum() == pytest.approx(1.0)
        seen_reserved += 1

    assert seen_reserved, "the tiny frame injects email-domain gaps"


def test_a_level_the_model_never_saw_lights_no_one_hot_column(
    tiny_explainer, complete_row: pd.DataFrame
):
    """``handle_unknown="ignore"`` is the locked behaviour; the explanation inherits it."""
    unseen = 0

    for feature in CATEGORICAL_FEATURES:
        _, column_names = _column_layout(tiny_explainer, feature)

        if any(name.endswith(RESERVED_CATEGORICAL_LEVEL) for name in column_names):
            continue

        gapped = complete_row.copy()
        gapped[feature] = None

        assert _one_hot_row(tiny_explainer, gapped, feature).sum() == 0.0
        unseen += 1

    assert unseen, "ProductCD and card6 have no gaps in the tiny frame"


def test_a_fully_gapped_row_still_produces_a_full_explanation(
    tiny_explainer, complete_row: pd.DataFrame
):
    gapped = complete_row.copy()

    for feature in CD_FEATURES:
        gapped[feature] = np.nan

    for feature in CATEGORICAL_FEATURES:
        gapped[feature] = None

    explained = tiny_explainer.explain(gapped)

    assert len(explained.contributions) == len(MODEL_FEATURES)
    assert all(np.isfinite(e.contribution) for e in explained.contributions)
    assert explained.base_value + sum(
        e.contribution for e in explained.contributions
    ) == pytest.approx(explained.raw_margin, abs=FLOAT32_ADDITIVITY_TOLERANCE)


def test_the_gapped_row_differs_from_the_gap_free_row(
    tiny_explainer, complete_row: pd.DataFrame
):
    """The missing-value path is not silently collapsed into the observed path."""
    gapped = complete_row.copy()
    gapped[CD_FEATURES[0]] = np.nan

    assert tiny_explainer.explain(gapped).to_dict() != tiny_explainer.explain(
        complete_row
    ).to_dict()



# --- input validation ----------------------------------------------------------


def test_a_non_dataframe_is_rejected(tiny_explainer):
    with pytest.raises(TypeError, match="single-row pandas DataFrame"):
        tiny_explainer.explain({MODEL_FEATURES[0]: 1.0})  # type: ignore[arg-type]


def test_two_rows_are_rejected(tiny_explainer, tiny_frame: pd.DataFrame):
    with pytest.raises(ValueError, match="exactly one transaction row"):
        tiny_explainer.explain(tiny_frame.iloc[[0, 1]])


def test_a_row_missing_contract_features_is_rejected(tiny_explainer, first_row):
    incomplete = first_row.drop(columns=[MODEL_FEATURES[0]])

    with pytest.raises(ValueError, match="missing features"):
        tiny_explainer.explain(incomplete)


def test_extra_columns_are_ignored_not_rejected(tiny_explainer, first_row):
    padded = first_row.assign(TransactionID=1, TransactionDT=1, isFraud=1)

    assert tiny_explainer.explain(padded).to_dict() == tiny_explainer.explain(
        first_row
    ).to_dict()


# --- builder validation --------------------------------------------------------


def test_a_non_pipeline_is_rejected():
    with pytest.raises(TypeError, match="sklearn Pipeline"):
        build_model_explainer("classifier")  # type: ignore[arg-type]


def test_a_pipeline_without_the_locked_steps_is_rejected():
    with pytest.raises(ValueError, match="missing required steps"):
        build_model_explainer(Pipeline(steps=[("scaler", StandardScaler())]))


def test_a_classifier_without_probabilities_is_rejected(tiny_frame: pd.DataFrame):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor()),
            ("classifier", _EstimatorWithoutProbabilities()),
        ]
    )
    pipeline.fit(tiny_frame[MODEL_FEATURES], tiny_frame[TARGET_COLUMN])

    with pytest.raises(TypeError, match="cannot produce probabilities"):
        build_model_explainer(pipeline)


def test_the_attribution_map_is_rebuilt_from_the_pipeline(tiny_explainer):
    rebuilt = build_feature_attribution_map(tiny_explainer.preprocessor)

    assert rebuilt.transformed_names == tiny_explainer.attribution.transformed_names
    assert rebuilt.source_features == tiny_explainer.attribution.source_features
