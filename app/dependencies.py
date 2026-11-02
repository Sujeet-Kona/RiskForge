"""Startup loading of the frozen artifacts, and the FastAPI dependencies.

The bundle is read once, when the application starts, and never again per
request. A failure to load is captured rather than raised so the process can
still answer ``/health`` and say why it cannot score — but nothing is trained,
reconstructed or approximated as a substitute.

The audit store is built here too, and is likewise never rebuilt per request.
Constructing it opens no connection, so a database that is down is discovered
when a decision is written rather than at startup: the process still serves
``/health`` and says which part of it is unavailable.

This is the only module in :mod:`app` that imports FastAPI's request machinery,
and the only one that reaches the database engine. :mod:`app.service` stays
framework-free so the same scoring path can be driven directly by the offline
parity check, with no database in sight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from fastapi import Request

from app.service import RiskScoringService
from db.config import DatabaseConfig
from db.repository import AuditRepository, AuditStore
from db.session import AuditDatabase
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
    audit: AuditStore
    failure_code: str | None


def build_audit_store() -> AuditStore:
    """The process-wide PostgreSQL audit store.

    Opens no connection: SQLAlchemy creates engines lazily, so this cannot fail
    because a database is unreachable.
    """
    database = AuditDatabase.create(DatabaseConfig.from_environment())

    return AuditRepository(database=database)


def load_service(
    artifact_path: Path,
    *,
    audit_store: AuditStore | None = None,
) -> StartupResult:
    """Load the artifact and build the scoring service exactly once."""
    audit = build_audit_store() if audit_store is None else audit_store

    try:
        artifact = load_artifact(artifact_path)
        service = RiskScoringService.from_artifact(artifact)
    except ArtifactNotFoundError as error:
        logger.error("serving artifact not found: %s", error)
        return StartupResult(service=None, audit=audit, failure_code=ARTIFACT_NOT_FOUND)
    except ArtifactError as error:
        logger.error("serving artifact rejected: %s", error)
        return StartupResult(service=None, audit=audit, failure_code=ARTIFACT_INVALID)

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

    return StartupResult(service=service, audit=audit, failure_code=None)


def release_resources(startup: StartupResult) -> None:
    """Close the connection pool at shutdown, if this process owns one."""
    if isinstance(startup.audit, AuditRepository):
        startup.audit.database.dispose()


def get_service(request: Request) -> RiskScoringService:
    """FastAPI dependency: the process-wide service built at startup."""
    startup: StartupResult = request.app.state.startup

    if startup.service is None:
        raise ArtifactUnavailableError(startup.failure_code or ARTIFACT_INVALID)

    return startup.service


def get_audit_store(request: Request) -> AuditStore:
    """FastAPI dependency: the process-wide audit store built at startup."""
    startup: StartupResult = request.app.state.startup

    return startup.audit
