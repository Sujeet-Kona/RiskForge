"""The serving layer's architectural guarantees, checked against the source.

Behavioural tests can show that the API did the right thing once; these show
that it cannot do the wrong thing. They are AST checks in the same style as
``test_explainability_discipline.py``: no training at request time, no second
feature pipeline, no later-phase infrastructure, no client-visible internals,
and no second copy of the frozen thresholds.
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

# Infrastructure that belongs to Phase 7 and later. None of it may appear while
# the service is still stateless.
LATER_PHASE_ROOTS = {
    "sqlalchemy",
    "alembic",
    "psycopg",
    "psycopg2",
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


# --- requirements 26 and 27: stateless, offline ------------------------------


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
    "directory", [POLICY_DIR, ML_DIR, EXPLAIN_DIR], ids=lambda p: p.name
)
def test_no_lower_layer_imports_the_serving_layer(directory: Path):
    for path in sorted(directory.rglob("*.py")):
        assert "app" not in _imported_roots(_tree(path)), path.name


@pytest.mark.parametrize("path", sorted(POLICY_DIR.glob("*.py")), ids=lambda p: p.name)
def test_the_policy_package_stays_independent_of_the_serving_stack(path: Path):
    forbidden = {"app", "fastapi", "starlette", "pydantic", "joblib"}

    assert not (_imported_roots(_tree(path)) & forbidden), path.name


def test_the_version_identifiers_are_the_documented_local_ones():
    """Project artifact identifiers, not registry versions."""
    from ml.models.artifact import FEATURE_VERSION, MODEL_VERSION

    assert MODEL_VERSION == "xgb-001"
    assert FEATURE_VERSION == "behavioral-v1"
