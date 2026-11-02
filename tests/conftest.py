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
    from policy.costs import EXPERIMENTAL_COST_ASSUMPTIONS
    from policy.decision import DecisionPolicy, PolicyConfig

    return DecisionPolicy(
        PolicyConfig(
            policy_version="policy-001",
            review_threshold=0.075,
            block_threshold=0.110,
            costs=EXPERIMENTAL_COST_ASSUMPTIONS,
        )
    )
