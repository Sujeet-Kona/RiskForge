"""Persistence for the frozen serving artifacts.

The trained pipeline and the fitted calibrator are the only two pieces of state
inference needs, and neither may be rebuilt when a request arrives. This module
gives them one durable home so the API can start from disk instead of depending
on a training script having just run.

Nothing here trains, tunes or derives anything: :func:`save_artifact` writes an
already-fitted bundle and :func:`load_artifact` reads it back, refusing the
bundle if it does not match the locked feature contract or the preprocessing
width the classifier was actually fitted on.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from ml.data.modeling import MODEL_FEATURES
from ml.models.calibration import FraudCalibrator
from ml.models.xgboost_model import (
    CLASSIFIER_STEP_NAME,
    PREPROCESSOR_STEP_NAME,
)


# Project artifact identifiers, not registry versions: RiskForge has no model
# registry, so these name the local bundle the serving path loads. They are
# stamped into the artifact at build time and returned verbatim by the API.
MODEL_VERSION = "xgb-001"
FEATURE_VERSION = "behavioral-v1"

ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_FILENAME = "riskforge_serving_artifact.joblib"

# A joblib bundle embeds the internal state of the fitted estimators, so it is
# only meaningful under the library versions that produced it. Loading one under
# a different scikit-learn or XGBoost can unpickle successfully and still score
# differently, which for a fraud decision is worse than failing outright.
_VERSION_GUARDED_LIBRARIES = ("scikit-learn", "xgboost")


class ArtifactError(RuntimeError):
    """The artifact exists but cannot be trusted for serving."""


class ArtifactNotFoundError(ArtifactError):
    """No artifact was found at the configured path."""


@dataclass(frozen=True)
class RiskForgeArtifact:
    """Everything the serving path needs, frozen at build time.

    ``pipeline`` is the locked preprocessing + XGBoost pipeline fitted on TRAIN
    only. ``calibrator`` is the locked isotonic mapping fitted on VALIDATION.
    The remaining fields are the identity and provenance of that bundle, kept
    with it so a response can always state which artifact produced it.
    """

    pipeline: Pipeline
    calibrator: FraudCalibrator
    model_version: str
    feature_version: str
    policy_version: str
    calibration_method: str
    review_threshold: float
    block_threshold: float
    scale_pos_weight: float
    n_transformed_columns: int
    feature_columns: tuple[str, ...]
    n_train_rows: int
    n_calibration_rows: int
    library_versions: dict[str, str]
    built_at_utc: str
    artifact_schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_artifact(self)

    @property
    def classifier(self) -> Any:
        return self.pipeline.named_steps[CLASSIFIER_STEP_NAME]

    def describe(self) -> dict[str, Any]:
        """A JSON-safe summary, deliberately excluding the fitted objects."""
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "policy_version": self.policy_version,
            "calibration_method": self.calibration_method,
            "review_threshold": self.review_threshold,
            "block_threshold": self.block_threshold,
            "scale_pos_weight": self.scale_pos_weight,
            "n_transformed_columns": self.n_transformed_columns,
            "n_features": len(self.feature_columns),
            "n_train_rows": self.n_train_rows,
            "n_calibration_rows": self.n_calibration_rows,
            "library_versions": dict(self.library_versions),
            "built_at_utc": self.built_at_utc,
        }


def current_library_versions() -> dict[str, str]:
    """Versions of the libraries whose internals the bundle embeds."""
    return {
        name: importlib.metadata.version(name)
        for name in _VERSION_GUARDED_LIBRARIES
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_artifact(artifact: RiskForgeArtifact, path: Path) -> Path:
    """Write the bundle to ``path``, creating parent directories."""
    if not isinstance(artifact, RiskForgeArtifact):
        raise TypeError("artifact must be a RiskForgeArtifact.")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, destination)

    return destination


def load_artifact(path: Path) -> RiskForgeArtifact:
    """Read a bundle written by :func:`save_artifact`.

    Fails clearly rather than degrading: a missing file, an unpicklable file, a
    foreign object, or a bundle whose contract no longer matches the code all
    raise an :class:`ArtifactError` subclass. No fallback model is ever
    substituted.
    """
    source = Path(path)

    if not source.exists():
        raise ArtifactNotFoundError(
            f"No serving artifact at {source}. Build it offline with "
            "`python scripts/build_serving_artifact.py`; the API never trains."
        )

    try:
        loaded = joblib.load(source)
    except Exception as error:
        raise ArtifactError(
            f"The serving artifact at {source} could not be read: "
            f"{type(error).__name__}."
        ) from error

    if not isinstance(loaded, RiskForgeArtifact):
        raise ArtifactError(
            f"The serving artifact at {source} is a "
            f"{type(loaded).__name__}, not a RiskForgeArtifact."
        )

    _guard_library_versions(loaded, source)

    return loaded


def _guard_library_versions(artifact: RiskForgeArtifact, source: Path) -> None:
    recorded = artifact.library_versions
    current = current_library_versions()

    mismatched = {
        name: (recorded.get(name), current[name])
        for name in _VERSION_GUARDED_LIBRARIES
        if recorded.get(name) != current[name]
    }

    if mismatched:
        detail = ", ".join(
            f"{name}: built with {built}, running {running}"
            for name, (built, running) in sorted(mismatched.items())
        )
        raise ArtifactError(
            f"The serving artifact at {source} was built under different "
            f"library versions ({detail}). Rebuild it offline rather than "
            "serving predictions from a bundle whose estimator internals may "
            "have changed meaning."
        )


def _validate_artifact(artifact: RiskForgeArtifact) -> None:
    """Reject a bundle that could not have come from the locked pipeline."""
    if artifact.artifact_schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError(
            f"Unsupported artifact schema version "
            f"{artifact.artifact_schema_version}; this code reads "
            f"{ARTIFACT_SCHEMA_VERSION}."
        )

    if not isinstance(artifact.pipeline, Pipeline):
        raise TypeError("pipeline must be a fitted sklearn Pipeline.")

    if PREPROCESSOR_STEP_NAME not in artifact.pipeline.named_steps:
        raise ArtifactError(
            "The artifact pipeline is missing the locked preprocessor step."
        )

    classifier_step = artifact.pipeline.named_steps.get(CLASSIFIER_STEP_NAME)

    if classifier_step is None:
        raise ArtifactError(
            "The artifact pipeline is missing the locked classifier step."
        )

    fitted_width = getattr(classifier_step, "n_features_in_", None)

    if fitted_width is None:
        raise ArtifactError(
            "The artifact classifier is not fitted, so it cannot serve scores."
        )

    if tuple(artifact.feature_columns) != tuple(MODEL_FEATURES):
        raise ArtifactError(
            "The artifact was built for a different feature contract "
            f"({len(artifact.feature_columns)} features); the locked contract "
            f"has {len(MODEL_FEATURES)}."
        )

    if artifact.n_transformed_columns != int(fitted_width):
        raise ArtifactError(
            f"The artifact records {artifact.n_transformed_columns} transformed "
            f"columns but its classifier was fitted on {int(fitted_width)}. "
            "Preprocessing and model disagree, so the bundle is not servable."
        )

    if not isinstance(artifact.calibrator, FraudCalibrator):
        raise TypeError("calibrator must be a FraudCalibrator.")

    if artifact.calibrator.estimator is None:
        raise ArtifactError(
            "The artifact calibrator is not fitted; the API must never fit "
            "calibration at request time."
        )

    if artifact.calibrator.method != artifact.calibration_method:
        raise ArtifactError(
            f"The artifact declares calibration method "
            f"{artifact.calibration_method!r} but carries a "
            f"{artifact.calibrator.method!r} estimator."
        )

    for name in ("review_threshold", "block_threshold"):
        value = float(getattr(artifact, name))

        if not 0.0 <= value <= 1.0:
            raise ArtifactError(f"{name} must be between 0 and 1.")

    if artifact.review_threshold >= artifact.block_threshold:
        raise ArtifactError(
            "review_threshold must be strictly less than block_threshold."
        )

    for name in ("model_version", "feature_version", "policy_version"):
        value = getattr(artifact, name)

        if not isinstance(value, str) or not value.strip():
            raise ArtifactError(f"{name} must be a non-blank string.")

    if not isinstance(artifact.scale_pos_weight, float):
        raise TypeError("scale_pos_weight must be a float.")

    if not artifact.scale_pos_weight > 0.0:
        raise ArtifactError("scale_pos_weight must be positive.")

    if artifact.n_train_rows <= 0 or artifact.n_calibration_rows <= 0:
        raise ArtifactError(
            "The artifact records an empty training or calibration partition."
        )
