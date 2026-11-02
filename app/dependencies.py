"""Startup loading of the frozen artifacts, and the FastAPI dependency.

The bundle is read once, when the application starts, and never again per
request. A failure to load is captured rather than raised so the process can
still answer ``/health`` and say why it cannot score — but nothing is trained,
reconstructed or approximated as a substitute.

This is the only module in :mod:`app` that imports FastAPI's request machinery.
:mod:`app.service` stays framework-free so the same scoring path can be driven
directly by the offline parity check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from fastapi import Request

from app.service import RiskScoringService
from ml.models.artifact import (
    ArtifactError,
    ArtifactNotFoundError,
    load_artifact,
)


logger = logging.getLogger("riskforge.serving")

ARTIFACT_NOT_FOUND = "artifact_not_found"
ARTIFACT_INVALID = "artifact_invalid"

# Client-facing wording, fixed per code. The underlying ArtifactError messages
# contain filesystem paths and are logged server-side only.
ARTIFACT_UNAVAILABLE_MESSAGE = (
    "The frozen serving artifact could not be loaded, so no transaction can be "
    "scored. This service never trains a model or fits a calibration at request "
    "time; build the artifact offline and restart."
)


class ArtifactUnavailableError(RuntimeError):
    """Raised per request when startup could not produce a scoring service."""

    def __init__(self, code: str) -> None:
        super().__init__(ARTIFACT_UNAVAILABLE_MESSAGE)
        self.code = code


@dataclass(frozen=True)
class StartupResult:
    service: RiskScoringService | None
    failure_code: str | None


def load_service(artifact_path: Path) -> StartupResult:
    """Load the artifact and build the scoring service exactly once."""
    try:
        artifact = load_artifact(artifact_path)
        service = RiskScoringService.from_artifact(artifact)
    except ArtifactNotFoundError as error:
        logger.error("serving artifact not found: %s", error)
        return StartupResult(service=None, failure_code=ARTIFACT_NOT_FOUND)
    except ArtifactError as error:
        logger.error("serving artifact rejected: %s", error)
        return StartupResult(service=None, failure_code=ARTIFACT_INVALID)

    logger.info(
        "loaded serving artifact model=%s feature=%s policy=%s "
        "transformed_columns=%d train_rows=%d calibration_rows=%s",
        artifact.model_version,
        artifact.feature_version,
        artifact.policy_version,
        artifact.n_transformed_columns,
        artifact.n_train_rows,
        f"{artifact.n_calibration_rows:,}",
    )

    return StartupResult(service=service, failure_code=None)


def get_service(request: Request) -> RiskScoringService:
    """FastAPI dependency: the process-wide service built at startup."""
    startup: StartupResult = request.app.state.startup

    if startup.service is None:
        raise ArtifactUnavailableError(startup.failure_code or ARTIFACT_INVALID)

    return startup.service
