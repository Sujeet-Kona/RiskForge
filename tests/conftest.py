from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from db.config import DATABASE_URL_ENVIRONMENT_VARIABLE, DEFAULT_DATABASE_URL
from ml.data.modeling import (
    CD_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_modeling_frame,
)
from ml.models.preprocessing import build_feature_preprocessor


REPO_ROOT = Path(__file__).resolve().parents[1]

TINY_ROWS = 600
TINY_SEED = 21

# The audit tests run against their own database so that a developer's local
# ``riskforge`` database, and anything in it, is never a test fixture.
AUDIT_TEST_DATABASE = "riskforge_test"


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
def client(tiny_artifact_path, audit_store):
    """A TestClient whose app boots from the on-disk tiny artifact.

    The audit store is injected rather than built from the environment, so no
    test that asserts the wire contract depends on a PostgreSQL server being up.
    """
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(
        create_app(artifact_path=tiny_artifact_path, audit_store=audit_store)
    ) as test_client:
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


# --- Phase 7: the audit store ------------------------------------------------


class InMemoryAuditStore:
    """An ``AuditStore`` double that keeps records in a list.

    Every test uses this unless it explicitly asks for the real database. That is
    not a way of skipping Phase 7 — the audit store's own behaviour is tested
    against PostgreSQL in ``test_db_models.py``, ``test_db_repository.py`` and
    ``test_app_audit_api.py``. What must not happen is the several hundred tests
    that assert the wire contract starting to depend on a database server, since
    the audit store may record a decision but may never change one.

    ``failure`` lets a test make the next operation fail the way the PostgreSQL
    repository fails, which is how the sanitized 503 path is exercised without
    taking a real database down.
    """

    def __init__(self) -> None:
        self.records: list = []
        self.failure: Exception | None = None
        self.ready: bool = True

    def reset(self) -> None:
        self.records = []
        self.failure = None
        self.ready = True

    def create_audit_record(self, record) -> int:
        if self.failure is not None:
            raise self.failure

        self.records.append(record)

        return len(self.records)

    def get_audit_record(self, decision_id: int):
        if self.failure is not None:
            raise self.failure

        if not 1 <= decision_id <= len(self.records):
            return None

        return self.records[decision_id - 1]

    def is_ready(self) -> bool:
        return self.ready


@pytest.fixture(scope="session")
def audit_store() -> Iterator[InMemoryAuditStore]:
    """The shared double, installed as the application's default audit store.

    Patched at session scope so that an application built inside a test body —
    not only the one behind the ``client`` fixture — also gets the double rather
    than a connection to whatever ``DATABASE_URL`` happens to point at.
    """
    import app.dependencies as dependencies

    store = InMemoryAuditStore()
    patch = pytest.MonkeyPatch()
    patch.setattr(dependencies, "build_audit_store", lambda: store)

    yield store

    patch.undo()


@pytest.fixture(autouse=True)
def clean_audit_store(
    audit_store: InMemoryAuditStore,
) -> Iterator[InMemoryAuditStore]:
    """Empty the shared double around every test, and expose it for assertions."""
    audit_store.reset()

    yield audit_store

    audit_store.reset()


@pytest.fixture
def audit_record():
    """A valid audit record, shaped exactly as the API persists a real decision."""
    from db.repository import AuditRecord

    return AuditRecord(
        transaction_id="3000001",
        risk_probability=0.09090909090909091,
        decision="REVIEW",
        reason_codes=[
            {
                "code": "RC-CARD-VELOCITY-1H",
                "title": "Unusual card velocity in the last hour",
                "type": "behavioral",
                "description": "This card transacted more often than usual.",
                "feature": "card_txn_count_1h",
                "contribution": 0.41,
                "direction": "risk_increasing",
                "n_transformed_columns": 1,
            }
        ],
        risk_decreasing_signals=[
            {
                "code": "RC-ADDRESS-VELOCITY-24H",
                "title": "Stable address activity over 24 hours",
                "type": "behavioral",
                "description": "This address showed no unusual burst.",
                "feature": "address_txn_count_24h",
                "contribution": -0.12,
                "direction": "risk_decreasing",
                "n_transformed_columns": 1,
            }
        ],
        decision_reason_code={
            "code": "DEC-REVIEW",
            "title": "Probability reached the review band",
            "type": "decision_rule",
            "description": "policy-001 sent this probability to REVIEW.",
        },
        decision_explanation=(
            "Calibrated risk probability 0.0909 is at or above the review "
            "threshold and below the block threshold, so policy-001 returns "
            "REVIEW."
        ),
        model_detail={
            "base_value": -3.1,
            "raw_margin": -2.3,
            "raw_model_probability": 0.0911,
            "attribution_space": "raw_margin",
            "attribution_space_note": "Contributions are in raw margin space.",
            "total_risk_increasing": 0.41,
            "total_risk_decreasing": -0.12,
        },
        policy_detail={"review_threshold": 0.075, "block_threshold": 0.110},
        scored_inputs={
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
        top_k=3,
        model_version="xgb-001",
        feature_version="behavioral-v1",
        policy_version="policy-001",
        feature_count=28,
        history_transactions_used=2,
    )


def _audit_test_url() -> str:
    """The audit test database on whichever server the service is pointed at."""
    configured = os.environ.get(DATABASE_URL_ENVIRONMENT_VARIABLE, "").strip()
    url = configured or DEFAULT_DATABASE_URL

    return url.rsplit("/", 1)[0] + "/" + AUDIT_TEST_DATABASE


def _create_test_database(url: str) -> None:
    """Create the test database, if it is missing and the role may create one."""
    from sqlalchemy import create_engine, text

    maintenance_url = url.rsplit("/", 1)[0] + "/postgres"
    database_name = url.rsplit("/", 1)[1]

    engine = create_engine(maintenance_url, isolation_level="AUTOCOMMIT")

    try:
        with engine.connect() as connection:
            present = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": database_name},
            ).scalar()

            if present is None:
                connection.execute(text(f'CREATE DATABASE "{database_name}"'))
    finally:
        engine.dispose()


@contextmanager
def _database_url_set(url: str) -> Iterator[None]:
    """Point ``DATABASE_URL`` at the test database for the duration of a block.

    Alembic resolves its URL through :mod:`db.config` and nowhere else, so the
    environment is the only way to aim a migration at the test database. It is
    restored on the way out, whatever happened inside.
    """
    configured = os.environ.get(DATABASE_URL_ENVIRONMENT_VARIABLE)
    os.environ[DATABASE_URL_ENVIRONMENT_VARIABLE] = url

    try:
        yield
    finally:
        if configured is None:
            del os.environ[DATABASE_URL_ENVIRONMENT_VARIABLE]
        else:
            os.environ[DATABASE_URL_ENVIRONMENT_VARIABLE] = configured


def _alembic_config():
    """Alembic configured from the committed ``alembic.ini``, by absolute path."""
    from alembic.config import Config

    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))

    return config


@pytest.fixture(scope="session")
def audit_database():
    """A real PostgreSQL database, migrated down to base and back up to head.

    The downgrade is part of the fixture on purpose: it proves the committed
    migration is reversible, and it guarantees the schema every database test
    sees was produced by Alembic in this session rather than left over from an
    earlier run.

    Skipped when no PostgreSQL is reachable. The skip is reported in the run
    summary, so the database tests can never pass silently without a database.
    """
    from alembic import command
    from sqlalchemy.exc import SQLAlchemyError

    from db.config import DatabaseConfig
    from db.session import AuditDatabase

    url = _audit_test_url()
    database = AuditDatabase.create(DatabaseConfig(url=url))

    try:
        _create_test_database(url)

        with database.engine.connect():
            pass
    except SQLAlchemyError as error:
        database.dispose()
        pytest.skip(f"no PostgreSQL available to the audit tests: {error}")

    with _database_url_set(url):
        command.downgrade(_alembic_config(), "base")
        command.upgrade(_alembic_config(), "head")

    yield database

    database.dispose()


@pytest.fixture
def alembic_config(audit_database):
    """Alembic aimed at the audit test database, for one test."""
    with _database_url_set(_audit_test_url()):
        yield _alembic_config()


@pytest.fixture
def audit_repository(audit_database):
    """The PostgreSQL repository, over a table emptied for this test alone."""
    from sqlalchemy import text

    from db.models import AUDIT_TABLE_NAME
    from db.repository import AuditRepository

    with audit_database.engine.begin() as connection:
        connection.execute(text(f"TRUNCATE TABLE {AUDIT_TABLE_NAME} RESTART IDENTITY"))

    return AuditRepository(database=audit_database)


@pytest.fixture
def persisted_client(tiny_artifact_path, audit_repository):
    """An application whose audit store is the real PostgreSQL repository."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(
        create_app(artifact_path=tiny_artifact_path, audit_store=audit_repository)
    ) as test_client:
        yield test_client
