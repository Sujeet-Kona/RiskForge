"""Structural guarantees for the explainability layer (brief requirements 12–16).

Every check here parses source as an AST rather than scanning substrings. That
is deliberate: these modules document the absence of exactly what is being
tested for ("no randomness", "never looks at a label"), and a naive substring
match trips on the module's own prose.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent

EXPLAIN_DIR = REPO_ROOT / "explain"
POLICY_DIR = REPO_ROOT / "policy"
TESTS_DIR = REPO_ROOT / "tests"

EXPLAIN_SOURCES = sorted(EXPLAIN_DIR.glob("*.py"))
POLICY_SOURCES = sorted(POLICY_DIR.glob("*.py"))

EXPLAINABILITY_TESTS = sorted(
    TESTS_DIR / name
    for name in (
        "conftest.py",
        "test_reason_codes.py",
        "test_feature_attribution.py",
        "test_model_explanation.py",
        "test_decision_explanation.py",
        "test_explanation.py",
    )
)

# Requirements 12 and 13: the explanation is a deterministic function of the
# fitted model. No generated prose, no remote call, no service stack.
LLM_AND_NETWORK_ROOTS = {
    "openai",
    "anthropic",
    "ollama",
    "langchain",
    "langgraph",
    "requests",
    "httpx",
    "urllib",
    "socket",
    "http",
}

SERVICE_STACK_ROOTS = {
    "fastapi",
    "starlette",
    "sqlalchemy",
    "alembic",
    "psycopg",
    "asyncpg",
    "pydantic",
    "redis",
    "kafka",
    "mlflow",
}

IO_CALLS = {"open", "urlopen", "post", "get_response", "connect", "execute"}

# Requirements 15 and 16: nothing in the explanation path may touch a label or
# the locked TEST partition. Identifiers only — see the module docstring.
LABEL_IDENTIFIERS = {
    "isFraud",
    "TARGET_COLUMN",
    "y_test",
    "y_train",
    "y_valid",
    "y_true",
    "y_pred",
    "labels",
    "target",
}

SPLIT_AND_EVALUATION_MODULES = {
    "ml.data.temporal_split",
    "ml.models.evaluation",
    "ml.models.calibration",
    "policy.evaluation",
}

TEST_PARTITION_ACCESS = {"read_parquet", "read_csv", "temporal_split", "test_frame"}


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text())


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    return roots


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
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
    """Names the code actually reads, writes or calls — never docstring prose."""
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


def test_the_explain_package_exists_and_is_scanned():
    assert EXPLAIN_SOURCES, "no explain sources found"
    assert POLICY_SOURCES, "no policy sources found"
    assert EXPLAINABILITY_TESTS, "no explainability test sources found"


# --- requirement 12: no LLM, no external call ---------------------------------


@pytest.mark.parametrize("path", EXPLAIN_SOURCES, ids=lambda p: p.name)
def test_no_explain_module_imports_an_llm_or_network_client(path: Path):
    assert not (_imported_roots(_tree(path)) & LLM_AND_NETWORK_ROOTS), path.name


@pytest.mark.parametrize("path", EXPLAIN_SOURCES, ids=lambda p: p.name)
def test_no_explain_module_performs_io_or_network_calls(path: Path):
    assert not (_called_names(_tree(path)) & IO_CALLS), path.name


def test_no_explain_module_builds_a_prompt_or_calls_a_model_api():
    """Deterministic templates only: no prompt strings, no chat/completion calls."""
    for path in EXPLAIN_SOURCES:
        identifiers = _referenced_identifiers(_tree(path))

        assert not (identifiers & {"prompt", "chat", "completions", "generate"}), path.name


# --- requirement 13: no web or database stack ---------------------------------


@pytest.mark.parametrize("path", EXPLAIN_SOURCES, ids=lambda p: p.name)
def test_no_explain_module_imports_the_service_stack(path: Path):
    assert not (_imported_roots(_tree(path)) & SERVICE_STACK_ROOTS), path.name


# --- requirement 14: the policy layer stays independent -----------------------


@pytest.mark.parametrize("path", POLICY_SOURCES, ids=lambda p: p.name)
def test_the_policy_package_does_not_import_explainability(path: Path):
    assert "explain" not in _imported_roots(_tree(path)), path.name


@pytest.mark.parametrize("path", POLICY_SOURCES, ids=lambda p: p.name)
def test_the_policy_package_does_not_import_the_model_stack(path: Path):
    forbidden = {"ml", "explain", "xgboost", "sklearn", "scipy", "shap"}

    assert not (_imported_roots(_tree(path)) & forbidden), path.name


def test_the_dependency_runs_explain_to_policy_and_never_backwards():
    imported = {
        module for path in EXPLAIN_SOURCES for module in _imported_modules(_tree(path))
    }

    assert "policy.decision" in imported


# --- requirements 15 and 16: no label access, no TEST partition ---------------


@pytest.mark.parametrize("path", EXPLAIN_SOURCES, ids=lambda p: p.name)
def test_no_explain_module_references_a_label(path: Path):
    assert not (_referenced_identifiers(_tree(path)) & LABEL_IDENTIFIERS), path.name


@pytest.mark.parametrize("path", EXPLAIN_SOURCES, ids=lambda p: p.name)
def test_no_explain_module_imports_the_split_or_evaluation_layers(path: Path):
    assert not (_imported_modules(_tree(path)) & SPLIT_AND_EVALUATION_MODULES), path.name


@pytest.mark.parametrize("path", EXPLAIN_SOURCES, ids=lambda p: p.name)
def test_the_explain_package_does_not_import_the_calibration_layer(path: Path):
    """Reason codes describe model signals, not the calibration mapping."""
    for module in _imported_modules(_tree(path)):
        assert "calibration" not in module, path.name


@pytest.mark.parametrize(
    "path", EXPLAINABILITY_TESTS, ids=lambda p: p.name
)
def test_no_explainability_test_touches_the_locked_test_partition(path: Path):
    """Requirement 15: the tests run on a synthetic frame, never on TEST rows."""
    identifiers = _referenced_identifiers(_tree(path))

    assert not (identifiers & TEST_PARTITION_ACCESS), path.name


def test_the_reason_code_taxonomy_is_pure_data():
    """The taxonomy must be readable without a fitted model or a dataset."""
    tree = _tree(EXPLAIN_DIR / "reason_codes.py")

    assert _imported_roots(tree) <= {"__future__", "dataclasses", "ml"}
    assert not (_called_names(tree) & IO_CALLS)


SMOKE_SCRIPT = REPO_ROOT / "scripts" / "explain_transactions.py"

# Call sites that would put a partition's rows (and therefore its labels) into
# the explanation path.
SCORING_CALLS = {
    "predict_fraud_probability",
    "fit_calibrator",
    "apply_calibrator",
    "explain_transaction",
    "decide_many",
    "fit_xgboost_model",
}


def test_the_smoke_script_exists():
    assert SMOKE_SCRIPT.is_file(), "scripts/explain_transactions.py is missing"


def test_the_smoke_script_never_feeds_the_test_partition_to_the_model():
    """Requirement 15: the demonstration runs on TRAIN and VALIDATION only.

    ``temporal_split`` inevitably returns the TEST frame, and printing its row
    count is what makes the omission explicit. What must not happen is any TEST
    row reaching scoring, calibration, the policy or the explainer.
    """
    source = SMOKE_SCRIPT.read_text()
    tree = ast.parse(source)

    offenders = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        name = node.func.id if isinstance(node.func, ast.Name) else None
        name = name or (
            node.func.attr if isinstance(node.func, ast.Attribute) else None
        )

        if name not in SCORING_CALLS:
            continue

        arguments = [
            ast.get_source_segment(source, argument) or ""
            for argument in [*node.args, *[keyword.value for keyword in node.keywords]]
        ]

        if any("split.test" in argument for argument in arguments):
            offenders.append(f"{name}(...{arguments})")

    assert not offenders, offenders


def test_the_smoke_script_selects_examples_from_validation():
    tree = _tree(SMOKE_SCRIPT)

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "select_examples" in called
    assert "temporal_split" in called
    assert "explain_transaction" in called


def test_the_test_partition_is_referenced_once_and_only_for_its_row_count():
    """``temporal_split`` returns TEST whether the script wants it or not.

    The only legitimate use is ``len(split.test)`` inside a print, so the
    omission is visible in the output. Anything else would mean TEST rows
    entered the explanation path.
    """
    tree = _tree(SMOKE_SCRIPT)

    uses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "test"
        and isinstance(node.value, ast.Name)
        and node.value.id == "split"
    ]

    assert len(uses) == 1
    assert isinstance(uses[0].ctx, ast.Load)

    enclosing = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and any(use in ast.walk(node) for use in uses)
    }

    assert enclosing == {"len", "print"}
