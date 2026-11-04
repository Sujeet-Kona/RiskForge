"""The provenance a trained model has to carry: code, environment, contract.

Three things decide what a fitted model means, and none of them is visible in the
model itself: the commit that produced the training code, the library versions
whose estimator internals are embedded in the bundle, and the exact feature
contract the bundle was fitted against. This module reads all three from their
authoritative sources rather than restating any of them.

The feature list in particular is never rebuilt here. It is imported from
:mod:`ml.data.modeling`, which is where the locked 28-feature contract lives, so
a recorded contract cannot drift from the one the model actually used.

When git cannot answer — a repository exported without ``.git``, git missing from
the environment — the value recorded is ``unavailable``. A placeholder SHA would
look like evidence of reproducibility while proving nothing.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from ml.data.modeling import (
    BEHAVIORAL_FEATURES,
    CATEGORICAL_FEATURES,
    CD_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    SPLIT_COLUMN,
    TARGET_COLUMN,
)
from ml.models.artifact import FEATURE_VERSION, MODEL_VERSION
from ml.split.temporal import temporal_split


UNAVAILABLE = "unavailable"
UNKNOWN = "unknown"

# The scientific stack whose versions change what a fitted bundle means. A
# scikit-learn or XGBoost upgrade can alter estimator internals that a pickled
# pipeline embeds, so these are recorded with every run. MLflow's own version is
# added by the module that does the logging, and Python's by ``sys``.
TRACKED_LIBRARIES = (
    "pandas",
    "numpy",
    "scikit-learn",
    "scipy",
    "xgboost",
    "pyarrow",
    "joblib",
    "shap",
)

# The temporal split is chronological and never separates equal timestamps. This
# string is recorded verbatim so a reader does not have to guess what "temporal"
# meant for this run.
SPLIT_STRATEGY = "chronological_by_unique_transaction_dt"

CONTRACT_SOURCE_MODULE = "ml.data.modeling"

GIT_TIMEOUT_SECONDS = 30


class LineageError(RuntimeError):
    """Provenance could not be established, so the run is not reproducible."""


@dataclass(frozen=True)
class Lineage:
    """Code and environment provenance for one training run."""

    commit_sha: str
    working_tree: str
    python_version: str
    library_versions: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "commit_sha": self.commit_sha,
            "working_tree": self.working_tree,
            "python_version": self.python_version,
            "library_versions": dict(self.library_versions),
        }


def library_versions(
    names: tuple[str, ...] = TRACKED_LIBRARIES,
) -> dict[str, str]:
    """Installed versions of the libraries the bundle's meaning depends on.

    A missing distribution is an error rather than a blank: a run recorded under
    an environment that lacks XGBoost could not have produced the model it
    claims, and silently logging ``not-installed`` would hide that.
    """
    versions: dict[str, str] = {}

    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise LineageError(
                f"{name} is required to establish training lineage but is not "
                "installed in this environment."
            ) from None

    return versions


def git_commit_sha(repo_root: Path) -> str:
    """The HEAD commit of ``repo_root``, or ``unavailable`` if git cannot say."""
    output = _run_git(repo_root, "rev-parse", "HEAD")

    return output if output else UNAVAILABLE


def git_working_tree_state(repo_root: Path) -> str:
    """``clean``, ``dirty``, or ``unknown``.

    Recorded because a dirty tree means the model was trained by code that is
    not in the commit the run points at. That is not a reason to refuse the run,
    but it is a reason never to claim the run is reproducible from the SHA alone.
    """
    output = _run_git(repo_root, "status", "--porcelain")

    if output is None:
        return UNKNOWN

    return "dirty" if output.strip() else "clean"


def _run_git(repo_root: Path, *arguments: str) -> str | None:
    """Run a read-only git command, returning ``None`` when git cannot answer."""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None

    return completed.stdout.strip()


def collect_lineage(repo_root: Path) -> Lineage:
    """Read commit, working-tree state, Python and library versions together."""
    return Lineage(
        commit_sha=git_commit_sha(repo_root),
        working_tree=git_working_tree_state(repo_root),
        python_version=sys.version.split()[0],
        library_versions=library_versions(),
    )


def feature_contract() -> dict[str, object]:
    """The locked feature contract, read from the module that defines it.

    Ordering is preserved and is the point of recording it: the preprocessing
    step and the classifier were fitted against this exact sequence, so a
    reordered list describes a different model.
    """
    return {
        "source_module": CONTRACT_SOURCE_MODULE,
        "feature_version": FEATURE_VERSION,
        "model_version": MODEL_VERSION,
        "feature_count": len(MODEL_FEATURES),
        "features": list(MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "behavioral_features": list(BEHAVIORAL_FEATURES),
        "cd_features": list(CD_FEATURES),
        "target_column": TARGET_COLUMN,
        "split_column": SPLIT_COLUMN,
    }


def feature_contract_text() -> str:
    """The ordered contract, one feature per line, for a human reading a run."""
    lines = [
        f"# feature_version={FEATURE_VERSION}",
        f"# model_version={MODEL_VERSION}",
        f"# source={CONTRACT_SOURCE_MODULE}",
        f"# count={len(MODEL_FEATURES)}",
        "# Ordering is significant: the pipeline was fitted against this sequence.",
    ]

    lines.extend(f"{index}\t{name}" for index, name in enumerate(MODEL_FEATURES))

    return "\n".join(lines) + "\n"


def split_strategy() -> dict[str, object]:
    """The split actually configured, read from the function's own defaults.

    The fractions come from :func:`temporal_split`'s signature rather than from
    constants restated here, so the recorded strategy is by construction the one
    the code applies.
    """
    parameters = inspect.signature(temporal_split).parameters

    return {
        "strategy": SPLIT_STRATEGY,
        "split_column": SPLIT_COLUMN,
        "train_fraction": float(parameters["train_fraction"].default),
        "validation_fraction": float(parameters["validation_fraction"].default),
    }
