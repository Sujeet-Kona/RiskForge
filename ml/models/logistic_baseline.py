from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from ml.models.preprocessing import build_categorical_preprocessor


@dataclass(frozen=True)
class LogisticBaseline:
    pipeline: Pipeline
    feature_columns: list[str]
    categorical_features: list[str]
    numeric_features: list[str]


def build_logistic_baseline() -> LogisticBaseline:
    """Build the first RiskForge fraud-classification baseline.

    Categorical handling is shared with the official XGBoost contract so both
    models agree on what a missing ``P_emaildomain`` means. Numeric features are
    additionally standardised, which the linear model requires and the tree model
    does not.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
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

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        solver="lbfgs",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return LogisticBaseline(
        pipeline=pipeline,
        feature_columns=MODEL_FEATURES.copy(),
        categorical_features=CATEGORICAL_FEATURES.copy(),
        numeric_features=NUMERIC_FEATURES.copy(),
    )


def fit_logistic_baseline(
    train: pd.DataFrame,
) -> LogisticBaseline:
    """Fit the baseline using training data only."""

    model = build_logistic_baseline()

    X_train = train[model.feature_columns]
    y_train = train[TARGET_COLUMN]

    model.pipeline.fit(X_train, y_train)

    return model


def predict_fraud_probability(
    model: LogisticBaseline,
    data: pd.DataFrame,
) -> pd.Series:
    """Return model-estimated fraud probabilities."""

    X = data[model.feature_columns]

    probabilities = model.pipeline.predict_proba(X)[:, 1]

    return pd.Series(
        probabilities,
        index=data.index,
        name="fraud_probability",
    )
