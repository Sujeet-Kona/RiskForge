"""The serving layer's architectural guarantees, checked against the source.

Behavioural tests can show that the API did the right thing once; these show
that it cannot do the wrong thing. They are AST checks in the same style as
``test_explainability_discipline.py``: no training at request time, no second
feature pipeline, no later-phase infrastructure, no client-visible internals,
no second copy of the frozen thresholds, and an audit store that the serving
layer reaches only through its boundary module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
APP_SOURCES = sorted(APP_DIR.glob("*.py"))
TEST_SOURCES = sorted((REPO_ROOT / "tests").glob("test_app_*.py"))

POLICY_DIR = REPO_ROOT / "policy"
ML_DIR = REPO_ROOT / "ml"
EXPLAIN_DIR = REPO_ROOT / "explain"
DB_DIR = REPO_ROOT / "db"
ALEMBIC_DIR = REPO_ROOT / "alembic"
TRACKING_DIR = REPO_ROOT / "tracking"
SCRIPTS_DIR = REPO_ROOT / "scripts"

DB_SOURCES = sorted(DB_DIR.glob("*.py"))
TRACKING_SOURCES = sorted(TRACKING_DIR.glob("*.py"))
SCRIPT_SOURCES = sorted(SCRIPTS_DIR.glob("*.py"))

# Infrastructure that is still out of scope. Phase 7 added an audit store and
# Phase 8 added experiment tracking, and nothing else: no cache, no queue, no
# worker, no orchestration, no object store, and no second database driver.
# ``mlflow`` stays on this list because it belongs to the offline tracking layer
# alone; the tests below are what confine it there.
LATER_PHASE_ROOTS = {
    "asyncpg",
    "redis",
    "kafka",
    "confluent_kafka",
    "celery",
    "mlflow",
    "kubernetes",
    "docker",
    "boto3",
    "pymongo",
}

# The offline tracking layer. Exactly one module is allowed to talk to MLflow, so
# that a prediction can never come to depend on a tracking store being reachable.
MLFLOW_ROOTS = {"mlflow"}

MLFLOW_CALLS = {
    "start_run",
    "log_model",
    "log_params",
    "log_metrics",
    "log_dict",
    "log_text",
    "register_model",
    "set_tracking_uri",
    "set_experiment",
}

# One feature from each family. A tracking module that retyped the contract
# instead of importing it would have to contain these, so their absence is what
# makes the recorded contract a read of ``ml.data.modeling`` rather than a copy.
FEATURE_CONTRACT_LITERALS = (
    "TransactionAmt",
    "amount_log1p",
    "card6",
    "P_emaildomain",
    "D15",
)

# The audit store is a dependency of the serving layer now, but only through its
# own boundary module. The ORM, the driver and the engine stay inside ``db``.
DATABASE_ROOTS = {"sqlalchemy", "psycopg", "psycopg2", "alembic"}

# Which ``app`` module may name which ``db`` module. ``dependencies.py`` is the
# only one allowed to build the store; everything else may only refer to the
# boundary type and the record it carries.
ALLOWED_DATABASE_IMPORTS = {
    "audit.py": {"db.repository"},
    "dependencies.py": {"db.config", "db.repository", "db.session"},
    "main.py": {"db.repository"},
}

# The audit store holds decisions. It must not hold, or import, any of the
# machinery that produces them.
MODEL_AND_SERVING_ROOTS = {
    "app",
    "explain",
    "ml",
    "scripts",
    "fastapi",
    "starlette",
    "xgboost",
    "sklearn",
    "scipy",
    "shap",
    "pandas",
    "numpy",
    "joblib",
}

AUTH_ROOTS = {"jwt", "jose", "passlib", "bcrypt", "authlib", "oauthlib", "itsdangerous"}

NETWORK_ROOTS = {"httpx", "requests", "urllib", "socket", "aiohttp", "http"}

RANDOMNESS_AND_LLM_ROOTS = {
    "random",
    "openai",
    "anthropic",
    "ollama",
    "langchain",
    "langgraph",
}

# Calls that would fit something. The API is only allowed to apply.
FITTING_CALLS = {
    "fit",
    "fit_transform",
    "fit_calibrator",
    "build_calibrator",
    "fit_xgboost_model",
    "build_xgboost_model",
    "build_feature_preprocessor",
    "partial_fit",
}

# Calls that would build a model rather than load one.
MODEL_CONSTRUCTION = {"XGBClassifier", "Pipeline", "ColumnTransformer", "IsotonicRegression"}

# A second feature pipeline would show up as pandas windowing in the app layer.
WINDOWING_CALLS = {"rolling", "expanding", "ewm", "shift", "groupby", "resample"}

LABEL_IDENTIFIERS = {"isFraud", "TARGET_COLUMN", "y_true", "label", "labels"}
TEST_PARTITION_ACCESS = {"read_parquet", "read_csv", "temporal_split", "test_frame"}

# The frozen thresholds have exactly one home.
THRESHOLD_LITERALS = ("0.075", "0.110")


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text())


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])

    return roots


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)

    return modules


def _called_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)

    return names


def _referenced_identifiers(tree: ast.AST) -> set[str]:
    identifiers: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            identifiers.add(node.arg)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)

    return identifiers


def test_the_app_package_exists_and_is_scanned():
    assert APP_SOURCES, "no app sources found"
    assert (APP_DIR / "main.py").is_file()


# --- requirements 19 and 20: nothing is fitted at request time ---------------


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_fits_anything(path: Path):
    assert not (_called_names(_tree(path)) & FITTING_CALLS), path.name


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_constructs_a_model(path: Path):
    assert not (_referenced_identifiers(_tree(path)) & MODEL_CONSTRUCTION), path.name


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_imports_a_training_script(path: Path):
    """Training stays offline; the API consumes an already-produced artifact."""
    modules = _imported_modules(_tree(path))

    assert not any(module.split(".")[0] == "scripts" for module in modules), path.name


def test_the_artifact_is_loaded_in_exactly_one_place():
    loaders = [
        path.name for path in APP_SOURCES if "load_artifact" in _called_names(_tree(path))
    ]

    assert loaders == ["dependencies.py"]


# --- one inference path ------------------------------------------------------


def test_the_app_layer_implements_no_feature_logic_of_its_own():
    """The behavioral features come from the offline function, or not at all."""
    tree = _tree(APP_DIR / "features.py")

    assert "build_behavioral_features" in _called_names(tree)
    assert not (_called_names(tree) & WINDOWING_CALLS)


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_reimplements_windowing(path: Path):
    assert not (_called_names(_tree(path)) & WINDOWING_CALLS), path.name


def test_scoring_goes_through_the_existing_inference_functions():
    called = _called_names(_tree(APP_DIR / "service.py"))

    assert {
        "build_serving_features",
        "predict_fraud_probability",
        "apply_calibrator",
        "explain_transaction",
    } <= called


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_calls_the_classifier_directly(path: Path):
    """``predict_proba`` would bypass the shared scoring entry point."""
    assert "predict_proba" not in _called_names(_tree(path)), path.name


def test_the_scoring_service_stays_database_free():
    """``service.score()`` is what the offline parity check drives.

    If the scoring path needed a database, the parity gate could no longer be run
    against the frozen artifact alone, and a decision would depend on something
    other than the model, the calibration and the policy.
    """
    modules = _imported_modules(_tree(APP_DIR / "service.py"))

    assert not any(module.split(".")[0] == "db" for module in modules)


# --- the audit store is a boundary, not a second dependency ------------------


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_imports_the_orm_or_a_driver(path: Path):
    """SQLAlchemy and psycopg stay inside ``db``; ``app`` names a boundary type."""
    assert not (_imported_roots(_tree(path)) & DATABASE_ROOTS), path.name


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_the_app_layer_reaches_the_database_through_a_narrow_door(path: Path):
    imported = {
        module
        for module in _imported_modules(_tree(path))
        if module.split(".")[0] == "db"
    }

    assert imported <= ALLOWED_DATABASE_IMPORTS.get(path.name, set()), path.name


def test_the_audit_store_is_built_in_exactly_one_place():
    """One construction site, so one place to point at a double in a test."""
    builders = [
        path.name
        for path in APP_SOURCES
        if "build_audit_store" in _called_names(_tree(path))
    ]

    assert builders == ["dependencies.py"]


def test_the_score_endpoint_records_the_decision_before_returning_it():
    """Persistence sits between the decision and the response, never after it.

    A write placed after the ``return`` would be dead code, and a response
    returned before the write would report a decision that might never be
    audited.
    """
    endpoint = next(
        node
        for node in ast.walk(_tree(APP_DIR / "main.py"))
        if isinstance(node, ast.FunctionDef) and node.name == "score"
    )

    recorded = [
        position
        for position, statement in enumerate(endpoint.body)
        if "record_decision" in _called_names(statement)
    ]
    returned = [
        position
        for position, statement in enumerate(endpoint.body)
        if isinstance(statement, ast.Return)
    ]

    assert recorded, "the score endpoint does not persist the decision"
    assert returned
    assert recorded[0] < returned[0]


@pytest.mark.parametrize("path", DB_SOURCES, ids=lambda p: p.name)
def test_the_audit_store_contains_no_model_or_serving_logic(path: Path):
    """The database records decisions; it cannot produce, alter or explain one."""
    assert not (_imported_roots(_tree(path)) & MODEL_AND_SERVING_ROOTS), path.name


@pytest.mark.parametrize("path", DB_SOURCES, ids=lambda p: p.name)
def test_no_audit_module_restates_the_frozen_thresholds(path: Path):
    source = path.read_text()

    for literal in THRESHOLD_LITERALS:
        assert literal not in source, f"{path.name} hardcodes {literal}"


# --- requirements 26 and 27: offline, and no network -------------------------


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_imports_later_phase_infrastructure(path: Path):
    assert not (_imported_roots(_tree(path)) & LATER_PHASE_ROOTS), path.name


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_imports_a_network_client(path: Path):
    assert not (_imported_roots(_tree(path)) & NETWORK_ROOTS), path.name


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_imports_authentication(path: Path):
    assert not (_imported_roots(_tree(path)) & AUTH_ROOTS), path.name


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_imports_randomness_or_an_llm(path: Path):
    """Requirement 24's precondition: the response cannot vary by chance."""
    assert not (_imported_roots(_tree(path)) & RANDOMNESS_AND_LLM_ROOTS), path.name


def test_fastapi_is_confined_to_the_two_modules_that_need_it():
    users = sorted(
        path.name
        for path in APP_SOURCES
        if _imported_roots(_tree(path)) & {"fastapi", "starlette"}
    )

    assert users == ["dependencies.py", "main.py"]


# --- requirement 25: no label access, no TEST partition ---------------------


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_references_a_label(path: Path):
    assert not (_referenced_identifiers(_tree(path)) & LABEL_IDENTIFIERS), path.name


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_reads_a_dataset_or_splits_it(path: Path):
    assert not (_referenced_identifiers(_tree(path)) & TEST_PARTITION_ACCESS), path.name


@pytest.mark.parametrize("path", TEST_SOURCES, ids=lambda p: p.name)
def test_no_serving_test_touches_the_locked_test_partition(path: Path):
    identifiers = _referenced_identifiers(_tree(path))

    assert not (identifiers & TEST_PARTITION_ACCESS), path.name
    assert not (identifiers & LABEL_IDENTIFIERS), path.name


# --- requirement 28 and 22: failures stay structured and terminal ----------


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_formats_a_traceback(path: Path):
    assert "traceback" not in _imported_roots(_tree(path)), path.name


def test_every_client_facing_error_message_is_a_module_level_constant():
    """No dynamic string is ever sent to a caller, so nothing internal can leak."""
    source = (APP_DIR / "main.py").read_text()
    tree = ast.parse(source)

    constants = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant)
    }
    constants.update(
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )

    messages = [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_error_response"
        for keyword in node.keywords
        if keyword.arg == "message"
    ]

    assert messages, "no error responses found"

    for message in messages:
        assert isinstance(message, ast.Name), ast.get_source_segment(source, message)
        assert message.id in constants, message.id


def test_a_missing_artifact_is_never_caught_and_replaced():
    """Requirement 22: fail clearly, never fall back to a substitute model."""
    source = (APP_DIR / "dependencies.py").read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue

        assert not (_called_names(node) & MODEL_CONSTRUCTION)
        assert "load_artifact" not in _called_names(node)


# --- the frozen thresholds have one home ------------------------------------


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_restates_the_frozen_thresholds(path: Path):
    """A second copy of 0.075 is a second place for policy-001 to drift."""
    source = path.read_text()

    for literal in THRESHOLD_LITERALS:
        assert literal not in source, f"{path.name} hardcodes {literal}"


def test_the_locked_policy_is_the_single_source_of_the_thresholds():
    from policy.locked import (
        LOCKED_BLOCK_THRESHOLD,
        LOCKED_POLICY_VERSION,
        LOCKED_REVIEW_THRESHOLD,
        build_locked_policy,
    )

    policy = build_locked_policy()

    assert LOCKED_POLICY_VERSION == "policy-001"
    assert LOCKED_REVIEW_THRESHOLD == 0.075
    assert LOCKED_BLOCK_THRESHOLD == 0.110
    assert policy.config.policy_version == LOCKED_POLICY_VERSION
    assert policy.config.review_threshold == LOCKED_REVIEW_THRESHOLD
    assert policy.config.block_threshold == LOCKED_BLOCK_THRESHOLD


# --- dependency direction ---------------------------------------------------


@pytest.mark.parametrize(
    "directory",
    [POLICY_DIR, ML_DIR, EXPLAIN_DIR, DB_DIR, ALEMBIC_DIR, TRACKING_DIR],
    ids=lambda p: p.name,
)
def test_no_lower_layer_imports_the_serving_layer(directory: Path):
    for path in sorted(directory.rglob("*.py")):
        assert "app" not in _imported_roots(_tree(path)), path.name


@pytest.mark.parametrize(
    "directory", [POLICY_DIR, ML_DIR, EXPLAIN_DIR], ids=lambda p: p.name
)
def test_no_model_layer_imports_the_audit_store(directory: Path):
    """Persistence is downstream of a decision, never an input to one."""
    for path in sorted(directory.rglob("*.py")):
        assert "db" not in _imported_roots(_tree(path)), path.name


@pytest.mark.parametrize("path", sorted(POLICY_DIR.glob("*.py")), ids=lambda p: p.name)
def test_the_policy_package_stays_independent_of_the_serving_stack(path: Path):
    forbidden = {"app", "db", "fastapi", "starlette", "pydantic", "joblib"}

    assert not (_imported_roots(_tree(path)) & forbidden), path.name


def test_the_version_identifiers_are_the_documented_local_ones():
    """Project artifact identifiers, not registry versions."""
    from ml.models.artifact import FEATURE_VERSION, MODEL_VERSION

    assert MODEL_VERSION == "xgb-001"
    assert FEATURE_VERSION == "behavioral-v1"


# --- Phase 8: MLflow is confined to the offline tracking layer ---------------

# Every directory that ships as product code. Tests are excluded: a test is
# allowed to drive MLflow in order to prove what a run recorded.
PRODUCT_DIRS = [
    APP_DIR,
    DB_DIR,
    ML_DIR,
    POLICY_DIR,
    EXPLAIN_DIR,
    ALEMBIC_DIR,
    SCRIPTS_DIR,
    TRACKING_DIR,
]

# The layers that must stay reachable without a tracking store existing.
NON_TRACKING_DIRS = [
    APP_DIR,
    DB_DIR,
    ML_DIR,
    POLICY_DIR,
    EXPLAIN_DIR,
    ALEMBIC_DIR,
]

MLFLOW_TRACKING_MODULE = Path("tracking") / "run.py"
TRACKED_TRAINING_SCRIPT = "train_tracked_model.py"

# Registering a version is bookkeeping. These are the calls that would decide
# what gets served, and nothing in this repository is allowed to make them.
PROMOTION_CALLS = {
    "set_registered_model_alias",
    "transition_model_version_stage",
    "delete_registered_model",
}


def _product_sources() -> list[Path]:
    return sorted(
        path
        for directory in PRODUCT_DIRS
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_the_tracking_package_exists_and_is_scanned():
    assert TRACKING_SOURCES, "no tracking sources found"
    assert (TRACKING_DIR / "run.py").is_file()


def test_mlflow_has_exactly_one_home_in_the_product_code():
    """One import site, so one place a runtime dependency could ever creep in."""
    users = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in _product_sources()
        if _imported_roots(_tree(path)) & MLFLOW_ROOTS
    )

    assert users == [MLFLOW_TRACKING_MODULE.as_posix()]


@pytest.mark.parametrize("directory", NON_TRACKING_DIRS, ids=lambda p: p.name)
def test_no_serving_or_model_layer_imports_mlflow(directory: Path):
    for path in sorted(directory.rglob("*.py")):
        assert not (_imported_roots(_tree(path)) & MLFLOW_ROOTS), path.name


@pytest.mark.parametrize("directory", NON_TRACKING_DIRS, ids=lambda p: p.name)
def test_no_serving_or_model_layer_imports_the_tracking_layer(directory: Path):
    """A prediction must not depend on experiment tracking being reachable."""
    for path in sorted(directory.rglob("*.py")):
        assert "tracking" not in _imported_roots(_tree(path)), path.name


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: p.name)
def test_no_app_module_makes_an_mlflow_call(path: Path):
    """The request path creates no run, logs no metric, queries no registry."""
    assert not (_called_names(_tree(path)) & MLFLOW_CALLS), path.name


def test_the_score_endpoint_never_touches_mlflow():
    endpoint = next(
        node
        for node in ast.walk(_tree(APP_DIR / "main.py"))
        if isinstance(node, ast.FunctionDef) and node.name == "score"
    )

    assert not (_called_names(endpoint) & MLFLOW_CALLS)
    assert not (_referenced_identifiers(endpoint) & {"mlflow", "tracking"})


def test_only_the_tracked_training_script_reaches_the_tracking_layer():
    users = sorted(
        path.name
        for path in SCRIPT_SOURCES
        if any(
            module.split(".")[0] == "tracking"
            for module in _imported_modules(_tree(path))
        )
    )

    assert users == [TRACKED_TRAINING_SCRIPT]


@pytest.mark.parametrize("path", TRACKING_SOURCES, ids=lambda p: p.name)
def test_the_tracking_layer_reads_the_feature_contract_rather_than_restating_it(path: Path):
    """A second copy of the 28 names is a second place for the contract to drift."""
    source = path.read_text()

    for literal in FEATURE_CONTRACT_LITERALS:
        assert literal not in source, f"{path.name} restates the feature {literal}"


def test_the_recorded_contract_is_imported_from_the_module_that_owns_it():
    imported = _imported_modules(_tree(TRACKING_DIR / "lineage.py"))

    assert "ml.data.modeling" in imported
    assert "ml.models.artifact" in imported


@pytest.mark.parametrize("path", TRACKING_SOURCES, ids=lambda p: p.name)
def test_no_tracking_module_restates_the_frozen_thresholds(path: Path):
    source = path.read_text()

    for literal in THRESHOLD_LITERALS:
        assert literal not in source, f"{path.name} hardcodes {literal}"


@pytest.mark.parametrize("path", TRACKING_SOURCES, ids=lambda p: p.name)
def test_the_tracking_layer_never_writes_or_reads_the_serving_artifact(path: Path):
    """Recording a run must not be able to change what the API serves."""
    called = _called_names(_tree(path))

    assert "save_artifact" not in called, path.name
    assert "load_artifact" not in called, path.name
    assert "fit_calibrator" not in called, path.name


@pytest.mark.parametrize("path", TRACKING_SOURCES, ids=lambda p: p.name)
def test_the_tracking_layer_never_promotes_a_model(path: Path):
    """A registered version is a label for inspection, never a deployment."""
    assert not (_called_names(_tree(path)) & PROMOTION_CALLS), path.name


@pytest.mark.parametrize("path", TRACKING_SOURCES, ids=lambda p: p.name)
def test_the_tracking_layer_stays_out_of_the_serving_and_audit_stacks(path: Path):
    forbidden = {"app", "db", "fastapi", "starlette", "shap"}

    assert not (_imported_roots(_tree(path)) & forbidden), path.name


@pytest.mark.parametrize("path", TRACKING_SOURCES, ids=lambda p: p.name)
def test_the_tracking_layer_opens_no_network_connection_of_its_own(path: Path):
    """MLflow owns whatever transport a store needs; this layer adds none.

    ``urllib.parse`` is deliberately not forbidden: parsing a tracking URI is
    string handling, and it is how the configuration refuses a remote one.
    """
    forbidden = {"requests", "httpx", "aiohttp", "socket", "http"}

    assert not (_imported_roots(_tree(path)) & forbidden), path.name
