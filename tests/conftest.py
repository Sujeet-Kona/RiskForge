from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml.data.modeling import (
    CD_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_modeling_frame,
)
from ml.models.preprocessing import build_feature_preprocessor


TINY_ROWS = 600
TINY_SEED = 21


def make_tiny_frame(n: int = TINY_ROWS, seed: int = TINY_SEED) -> pd.DataFrame:
    """A small modeling frame that exercises the real 28-feature contract.

    Synthetic on purpose: the explainability tests have to run without the
    gitignored IEEE-CIS dataset. It is assembled through the real
    ``build_modeling_frame`` and fitted through the real §14 preprocessor, so
    the contract under test is the production one rather than a stand-in.

    Labels are driven by a few of the features, which is what makes the
    resulting TreeSHAP values non-trivial in both directions.
    """
    rng = np.random.default_rng(seed)

    transaction_id = np.arange(3_000_000, 3_000_000 + n)
    transaction_dt = np.cumsum(rng.integers(1, 60, n)) + 86_400

    card_txn_count_1h = rng.poisson(0.6, n).astype(float)
    card_txn_count_24h = card_txn_count_1h + rng.poisson(2.0, n)
    address_txn_count_1h = rng.poisson(0.4, n).astype(float)
    address_txn_count_24h = address_txn_count_1h + rng.poisson(1.5, n)

    transaction_amt = rng.gamma(2.0, 45.0, n)

    product_levels = np.array(["W", "C", "R", "S", "H"])
    product_cd = product_levels[rng.integers(0, len(product_levels), n)]

    pressure = (
        0.55 * card_txn_count_1h
        + 0.012 * transaction_amt
        + 0.35 * (product_cd == "W")
        - 0.20 * address_txn_count_24h
        - 2.35
    )
    labels = (rng.random(n) < 1.0 / (1.0 + np.exp(-pressure))).astype(int)

    behavioral = pd.DataFrame(
        {
            "TransactionID": transaction_id,
            "TransactionDT": transaction_dt,
            "TransactionAmt": transaction_amt,
            "amount_log1p": np.log1p(transaction_amt),
            "transaction_hour_bucket": (transaction_dt // 3600) % 24,
            "transaction_day_mod7": (transaction_dt // 86400) % 7,
            "card_txn_count_1h": card_txn_count_1h,
            "card_txn_count_24h": card_txn_count_24h,
            "card_has_history_1h": card_txn_count_1h > 0,
            "card_has_history_24h": card_txn_count_24h > 0,
            "card_amount_sum_24h": transaction_amt * card_txn_count_24h,
            "address_txn_count_1h": address_txn_count_1h,
            "address_txn_count_24h": address_txn_count_24h,
            "address_has_history_1h": address_txn_count_1h > 0,
            "address_amount_sum_24h": transaction_amt * address_txn_count_24h,
        }
    )

    transactions = pd.DataFrame(
        {
            "TransactionID": transaction_id,
            "ProductCD": product_cd,
            "card6": np.array(["debit", "credit"])[rng.integers(0, 2, n)],
            "P_emaildomain": _with_missing(rng, n, ["gmail.com", "yahoo.com", "outlook.com"]),
            "R_emaildomain": _with_missing(rng, n, ["gmail.com", "protonmail.com"]),
        }
    )

    for feature in CD_FEATURES:
        column = rng.integers(0, 6, n).astype(float) + rng.gamma(1.5, 3.0, n)
        # Absence is signal in the real dataset, so the C/D family has to carry
        # gaps for the missing-indicator columns to exist at all.
        column[rng.random(n) < 0.25] = np.nan
        transactions[feature] = column

    frame = build_modeling_frame(
        behavioral=behavioral,
        transaction_features=transactions,
        labels=pd.DataFrame({"TransactionID": transaction_id, TARGET_COLUMN: labels}),
    )

    return frame.data


def _with_missing(rng: np.random.Generator, n: int, levels: list[str]) -> np.ndarray:
    values = np.array(levels, dtype=object)[rng.integers(0, len(levels), n)]
    values[rng.random(n) < 0.30] = None

    return values


@pytest.fixture(scope="session")
def tiny_frame() -> pd.DataFrame:
    return make_tiny_frame()


@pytest.fixture(scope="session")
def tiny_pipeline(tiny_frame: pd.DataFrame) -> Pipeline:
    """The locked pipeline shape, fitted on the tiny frame.

    Deliberately not the locked hyperparameters: this fixture exists to give the
    explainability tests a real fitted booster to decompose, not to reproduce
    the production model. Nothing here is a surrogate for it.
    """
    from xgboost import XGBClassifier

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=40,
                    max_depth=3,
                    learning_rate=0.3,
                    random_state=42,
                    n_jobs=1,
                    tree_method="hist",
                ),
            ),
        ]
    )

    pipeline.fit(tiny_frame[MODEL_FEATURES], tiny_frame[TARGET_COLUMN])

    return pipeline


@pytest.fixture(scope="session")
def tiny_explainer(tiny_pipeline: Pipeline):
    from explain.model_explanation import build_model_explainer

    return build_model_explainer(tiny_pipeline)


@pytest.fixture(scope="session")
def tiny_policy():
    """The frozen Phase 4 policy, with its real locked thresholds."""
    from policy.locked import build_locked_policy

    return build_locked_policy()


@pytest.fixture(scope="session")
def tiny_calibrator(tiny_frame: pd.DataFrame, tiny_pipeline: Pipeline):
    """An isotonic mapping fitted on the tiny frame's own held-out-style scores.

    Deliberately not the locked Phase 3 calibrator: this exists so the serving
    tests have a real fitted mapping to apply, not to reproduce production
    calibration.
    """
    from ml.models.calibration import fit_calibrator

    raw = tiny_pipeline.predict_proba(tiny_frame[MODEL_FEATURES])[:, 1]

    return fit_calibrator(
        "isotonic",
        pd.Series(raw, index=tiny_frame.index),
        tiny_frame[TARGET_COLUMN],
    )


@pytest.fixture(scope="session")
def tiny_artifact(tiny_pipeline: Pipeline, tiny_calibrator):
    """A complete serving bundle around the tiny fitted pipeline."""
    from ml.models.artifact import (
        FEATURE_VERSION,
        MODEL_VERSION,
        RiskForgeArtifact,
        current_library_versions,
        utc_now_iso,
    )

    classifier = tiny_pipeline.named_steps["classifier"]

    return RiskForgeArtifact(
        pipeline=tiny_pipeline,
        calibrator=tiny_calibrator,
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        policy_version="policy-001",
        calibration_method="isotonic",
        review_threshold=0.075,
        block_threshold=0.110,
        scale_pos_weight=1.0,
        n_transformed_columns=int(classifier.n_features_in_),
        feature_columns=tuple(MODEL_FEATURES),
        n_train_rows=TINY_ROWS,
        n_calibration_rows=TINY_ROWS,
        library_versions=current_library_versions(),
        built_at_utc=utc_now_iso(),
    )


@pytest.fixture(scope="session")
def tiny_artifact_path(tiny_artifact, tmp_path_factory):
    """The bundle written to disk and read back, as the API reads it.

    Going through save/load rather than handing the in-memory object to the
    service is what makes the fixtures exercise the real artifact contract.
    """
    from ml.models.artifact import ARTIFACT_FILENAME, save_artifact

    directory = tmp_path_factory.mktemp("serving-artifact")
    path = directory / ARTIFACT_FILENAME

    save_artifact(tiny_artifact, path)

    return path


@pytest.fixture(scope="session")
def loaded_artifact(tiny_artifact_path):
    from ml.models.artifact import load_artifact

    return load_artifact(tiny_artifact_path)


@pytest.fixture(scope="session")
def serving_service(loaded_artifact):
    from app.service import RiskScoringService

    return RiskScoringService.from_artifact(loaded_artifact)


@pytest.fixture(scope="session")
def client(tiny_artifact_path):
    """A TestClient whose app boots from the on-disk tiny artifact."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(artifact_path=tiny_artifact_path)) as test_client:
        yield test_client


@pytest.fixture
def score_payload() -> dict:
    """A minimal well-formed scoring request, as a caller would send it."""
    return {
        "transaction": {
            "TransactionID": "3000001",
            "TransactionDT": 200_000,
            "TransactionAmt": 117.5,
            "card1": 13926,
            "addr1": 299.0,
            "addr2": 87.0,
            "ProductCD": "W",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "R_emaildomain": None,
            "C1": 1.0,
            "C2": 1.0,
            "C11": 2.0,
            "C13": 1.0,
            "D2": None,
            "D3": None,
            "D4": 0.0,
            "D8": None,
            "D10": 1.0,
            "D11": 1.0,
            "D15": 0.0,
        },
        "history": [
            {
                "TransactionDT": 199_000,
                "TransactionAmt": 42.0,
                "card1": 13926,
                "addr1": 299.0,
                "addr2": 87.0,
            },
            {
                "TransactionDT": 150_000,
                "TransactionAmt": 18.25,
                "card1": 13926,
                "addr1": 299.0,
                "addr2": 87.0,
            },
        ],
    }
