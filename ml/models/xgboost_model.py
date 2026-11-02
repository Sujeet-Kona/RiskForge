from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml.data.modeling import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from ml.models.preprocessing import build_feature_preprocessor


# The two step names are part of the locked pipeline's identity: the serving
# artifact, the explainer and every consumer address the fitted objects through
# them. They are defined here because this module builds the pipeline.
PREPROCESSOR_STEP_NAME = "preprocessor"
CLASSIFIER_STEP_NAME = "classifier"


@dataclass(frozen=True)
class XGBoostFraudModel:
    model: Pipeline
    feature_columns: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    scale_pos_weight: float


def build_xgboost_model(train: pd.DataFrame) -> XGBoostFraudModel:
    """Build the official 28-feature RiskForge fraud model.

    Preprocessing and the classifier are returned as one unfitted pipeline, so
    a single ``fit`` on the TRAIN partition establishes both the imputation
    statistics / one-hot vocabulary and the trees. Validation and test data can
    therefore never influence model construction.

    Class imbalance is estimated from the training partition only.
    """
    if train.empty:
        raise ValueError("Training dataframe must not be empty.")

    missing = [
        column
        for column in [*MODEL_FEATURES, TARGET_COLUMN]
        if column not in train.columns
    ]
    if missing:
        raise ValueError(f"Training dataframe is missing required columns: {missing}")

    y_train = train[TARGET_COLUMN]

    if y_train.isna().any():
        raise ValueError("Training target contains missing values.")

    if not y_train.isin([0, 1]).all():
        raise ValueError("Training target must contain only 0/1 values.")

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)

    if positives == 0:
        raise ValueError("Training data contains no positive fraud cases.")

    if negatives == 0:
        raise ValueError("Training data contains no negative cases.")

    scale_pos_weight = negatives / positives

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    pipeline = Pipeline(
        steps=[
            (PREPROCESSOR_STEP_NAME, build_feature_preprocessor()),
            (CLASSIFIER_STEP_NAME, classifier),
        ]
    )

    return XGBoostFraudModel(
        model=pipeline,
        feature_columns=MODEL_FEATURES.copy(),
        categorical_features=CATEGORICAL_FEATURES.copy(),
        numeric_features=NUMERIC_FEATURES.copy(),
        scale_pos_weight=scale_pos_weight,
    )


def fit_xgboost_model(train: pd.DataFrame) -> XGBoostFraudModel:
    """Fit preprocessing and XGBoost using the training partition only."""
    model = build_xgboost_model(train)

    X_train = train[model.feature_columns]
    y_train = train[TARGET_COLUMN]

    model.model.fit(X_train, y_train)

    return model


def predict_fraud_probability(
    model: XGBoostFraudModel,
    data: pd.DataFrame,
) -> pd.Series:
    """Return XGBoost fraud scores.

    These are model scores/probabilities, not calibrated probabilities.
    Calibration is a later RiskForge phase.
    """
    missing = [
        column for column in model.feature_columns if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Prediction dataframe is missing features: {missing}")

    probabilities = model.model.predict_proba(
        data[model.feature_columns]
    )[:, 1]

    return pd.Series(
        probabilities,
        index=data.index,
        name="fraud_probability",
    )
