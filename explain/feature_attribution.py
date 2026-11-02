from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from ml.data.modeling import MODEL_FEATURES


NUMERIC_TRANSFORMER_NAME = "numeric"
CATEGORICAL_TRANSFORMER_NAME = "categorical"
ONE_HOT_STEP_NAME = "one_hot"

CONTRIBUTION_NAME = "contribution"


@dataclass(frozen=True)
class FeatureAttributionMap:
    """Maps every transformed model column back to its contract feature.

    The locked XGBoost model never sees the 28 contract features directly. The
    fitted ``ColumnTransformer`` expands them into 162 columns — 24 numeric
    values, 7 numeric missing indicators and 131 one-hot levels — and the trees
    split on those. Any per-transaction explanation therefore has to be summed
    back over the columns that belong to each contract feature before it means
    anything to a reader.

    The mapping is derived from the *fitted* estimators, not from parsing
    feature-name strings, so it stays correct whatever the one-hot vocabulary
    turns out to be. Because SHAP values are exactly additive over the columns
    the model actually consumes, summing a feature's columns is an exact
    attribution rather than an approximation.
    """

    contract_features: list[str]
    transformed_names: list[str]
    source_features: list[str]
    columns_by_feature: dict[str, list[int]]

    @property
    def n_transformed_columns(self) -> int:
        return len(self.transformed_names)


def build_feature_attribution_map(
    preprocessor: ColumnTransformer,
) -> FeatureAttributionMap:
    """Build the transformed-column → contract-feature mapping.

    ``preprocessor`` must be the *fitted* preprocessor taken from the locked
    pipeline. Nothing here is fitted, estimated or inferred from data.
    """
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError("preprocessor must be a fitted sklearn ColumnTransformer.")

    fitted_blocks = getattr(preprocessor, "transformers_", None)

    if not fitted_blocks:
        raise ValueError(
            "preprocessor is not fitted; it has no transformers_ attribute."
        )

    block_names_present = [name for name, _, _ in fitted_blocks]

    for required in (NUMERIC_TRANSFORMER_NAME, CATEGORICAL_TRANSFORMER_NAME):
        if required not in block_names_present:
            raise ValueError(
                f"preprocessor has no fitted {required!r} block; found "
                f"{block_names_present}."
            )

    source_features: list[str] = []
    expected_names: list[str] = []

    for name, transformer, block_columns in fitted_blocks:
        columns = list(block_columns)

        if name == NUMERIC_TRANSFORMER_NAME:
            sources = _numeric_sources(transformer, columns)
        elif name == CATEGORICAL_TRANSFORMER_NAME:
            sources = _categorical_sources(transformer, columns)
        else:
            raise ValueError(
                f"Unexpected preprocessor block {name!r}; the attribution map "
                "only knows how to expand the numeric and categorical blocks."
            )

        names = list(transformer.get_feature_names_out(columns))

        if len(names) != len(sources):
            raise ValueError(
                f"Block {name!r} produced {len(names)} transformed columns but "
                f"the attribution map assigned {len(sources)}."
            )

        source_features.extend(sources)
        expected_names.extend(f"{name}__{suffix}" for suffix in names)

    transformed_names = list(preprocessor.get_feature_names_out())

    if transformed_names != expected_names:
        raise ValueError(
            "The attribution map's column order does not match the "
            "preprocessor's own feature names; refusing to explain against a "
            "mapping that may be misaligned."
        )

    unknown = sorted(set(source_features) - set(MODEL_FEATURES))

    if unknown:
        raise ValueError(
            f"Transformed columns traced back to features outside the modeling "
            f"contract: {unknown}."
        )

    uncovered = sorted(set(MODEL_FEATURES) - set(source_features))

    if uncovered:
        raise ValueError(
            f"Contract features with no transformed column: {uncovered}."
        )

    columns_by_feature: dict[str, list[int]] = {
        feature: [] for feature in MODEL_FEATURES
    }

    for position, feature in enumerate(source_features):
        columns_by_feature[feature].append(position)

    return FeatureAttributionMap(
        contract_features=list(MODEL_FEATURES),
        transformed_names=transformed_names,
        source_features=source_features,
        columns_by_feature=columns_by_feature,
    )


def _numeric_sources(
    imputer: object,
    columns: list[str],
) -> list[str]:
    """Expand the numeric block: imputed values, then missing indicators.

    ``SimpleImputer(add_indicator=True)`` emits every input column followed by
    one indicator column per feature that had missing values during fitting.
    ``indicator_.features_`` holds those positions, so the indicators are
    attributed to the feature they describe rather than treated as new
    features.
    """
    if not hasattr(imputer, "get_feature_names_out"):
        raise TypeError("The numeric block is not a fitted sklearn transformer.")

    sources = list(columns)

    indicator = getattr(imputer, "indicator_", None)

    if indicator is not None:
        positions = getattr(indicator, "features_", None)

        if positions is None:
            raise ValueError(
                "The numeric missing indicator is present but not fitted."
            )

        for position in positions:
            if not 0 <= int(position) < len(columns):
                raise ValueError(
                    f"Missing indicator points at column {position}, outside "
                    f"the numeric block of width {len(columns)}."
                )
            sources.append(columns[int(position)])

    return sources


def _categorical_sources(
    pipeline: object,
    columns: list[str],
) -> list[str]:
    """Expand the categorical block using the fitted one-hot vocabulary.

    With no level dropping or infrequent-level collapsing, each input feature
    owns exactly ``len(categories_[i])`` contiguous output columns, in input
    order. Anything that would break that arithmetic is rejected rather than
    guessed at.
    """
    named_steps = getattr(pipeline, "named_steps", None)

    if named_steps is None or ONE_HOT_STEP_NAME not in named_steps:
        raise ValueError(
            f"The categorical block has no {ONE_HOT_STEP_NAME!r} step; found "
            f"{sorted(named_steps) if named_steps else 'no pipeline'}."
        )

    encoder = named_steps[ONE_HOT_STEP_NAME]
    categories = getattr(encoder, "categories_", None)

    if categories is None:
        raise ValueError("The one-hot encoder is not fitted.")

    for setting in ("drop", "min_frequency", "max_categories"):
        value = getattr(encoder, setting, None)

        if value is not None:
            raise ValueError(
                f"OneHotEncoder({setting}={value!r}) would make the number of "
                "output columns per feature unequal to len(categories_[i]); "
                "the attribution map refuses to guess the layout."
            )

    if len(categories) != len(columns):
        raise ValueError(
            f"The one-hot encoder was fitted on {len(categories)} columns but "
            f"the categorical block declares {len(columns)}."
        )

    sources: list[str] = []

    for feature, levels in zip(columns, categories):
        sources.extend([feature] * len(levels))

    return sources


def group_contributions(
    attribution: FeatureAttributionMap,
    contributions: np.ndarray,
) -> pd.Series:
    """Sum per-column contributions into one value per contract feature.

    This is exact, not approximate: the contributions are additive over the
    columns the model consumes, so a feature's total is the sum of the columns
    that encode it — including its missing indicator and all of its one-hot
    levels.
    """
    values = np.asarray(contributions, dtype=float).ravel()

    if values.size != attribution.n_transformed_columns:
        raise ValueError(
            f"Expected {attribution.n_transformed_columns} transformed-column "
            f"contributions, got {values.size}."
        )

    if not np.isfinite(values).all():
        raise ValueError("Contributions contain missing or non-finite values.")

    totals = np.array(
        [
            values[attribution.columns_by_feature[feature]].sum()
            for feature in attribution.contract_features
        ],
        dtype=float,
    )

    return pd.Series(
        totals,
        index=pd.Index(attribution.contract_features, name="feature"),
        name=CONTRIBUTION_NAME,
    )
