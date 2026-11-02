from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from ml.data.modeling import CATEGORICAL_FEATURES, NUMERIC_FEATURES


CATEGORICAL_MISSING_TOKEN = "__MISSING__"


def _normalize_missing(values: pd.DataFrame) -> pd.DataFrame:
    """Map NA-like sentinels to ``np.nan``; ``SimpleImputer`` ignores a bare ``None``."""

    return values.mask(values.isna(), np.nan)


def build_categorical_preprocessor() -> Pipeline:
    """Build the §14 categorical block, shared by every RiskForge model.

    Absence is signal in this dataset, so missing categorical values become the
    reserved ``__MISSING__`` level and are one-hot encoded alongside the observed
    levels rather than being dropped or backfilled with a frequent value.
    """

    return Pipeline(
        steps=[
            (
                "normalize_missing",
                FunctionTransformer(
                    _normalize_missing,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=CATEGORICAL_MISSING_TOKEN,
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )


def build_feature_preprocessor() -> ColumnTransformer:
    """Build the official preprocessing for the 28-feature contract.

    Numeric gaps (the C/D family) become a train median fill plus an explicit
    missing indicator; categorical gaps are handled by
    :func:`build_categorical_preprocessor`.

    The returned transformer is deliberately unfitted. It must be fit on the
    TRAIN partition only, so validation and test data can never influence the
    imputation statistics or the one-hot vocabulary.
    """

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                build_categorical_preprocessor(),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
