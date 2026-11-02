"""The RiskForge inference application.

Two endpoints, one scoring path:

``GET /health``
    Liveness, the identity of the loaded artifact, and whether the audit store is
    reachable. Performs no inference.

``POST /v1/risk/score``
    Scores one transaction against caller-supplied strictly-prior history,
    records the resulting decision in the PostgreSQL audit store, and returns the
    calibrated probability, the frozen policy decision, the Phase 5 reason codes
    and the version identifiers of everything that produced them.

The service holds no transaction history of its own. Behavioral features are
computed from the ``history`` array the caller sends; the audit store is written
after a decision has been made and is never read back as feature history. A
decision that cannot be recorded is not returned — see :mod:`app.audit`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.audit import (
    AUDIT_UNAVAILABLE_MESSAGE,
    AuditUnavailableError,
    record_decision,
)
from app.config import DEFAULT_ARTIFACT_PATH, HISTORY_NOTE
from app.dependencies import (
    ARTIFACT_UNAVAILABLE_MESSAGE,
    ArtifactUnavailableError,
    get_audit_store,
    get_service,
    load_service,
    release_resources,
)
from app.schemas import (
    ErrorOut,
    ErrorResponse,
    HealthResponse,
    RiskScoreRequest,
    RiskScoreResponse,
)
from app.service import RiskScoringService, to_response
from db.repository import AuditStore


logger = logging.getLogger("riskforge.serving")

API_TITLE = "RiskForge Inference API"
API_VERSION = "phase-7"

HEALTH_ROUTE = "/health"
SCORE_ROUTE = "/v1/risk/score"

DATABASE_READY = "ready"
DATABASE_UNAVAILABLE = "unavailable"

INFERENCE_FAILED = "inference_failed"
INFERENCE_FAILED_MESSAGE = (
    "Scoring failed for this transaction. No risk probability and no decision "
    "are being returned; a failure is never reported as a score."
)

API_DESCRIPTION = f"""
Scores a transaction with the locked RiskForge pipeline: the 28-feature
point-in-time contract, the XGBoost model fitted on TRAIN only, the isotonic
calibration fitted on VALIDATION, the frozen `policy-001` thresholds, and the
Phase 5 reason codes.

{HISTORY_NOTE}

Every decision that is returned has first been written to a PostgreSQL audit
store, so it can be reconstructed later. The database records decisions; it never
supplies a feature. A decision that cannot be recorded is reported as a 503
failure rather than returned unaudited.

The model, calibration and preprocessing are loaded once at startup from a frozen
artifact. This service never trains, never fits a calibrator and never falls back
to a substitute model: if the artifact cannot be loaded, scoring returns 503.
"""


def create_app(
    *,
    artifact_path: Path | None = None,
    audit_store: AuditStore | None = None,
) -> FastAPI:
    """Build the application, loading its artifacts once during startup.

    ``audit_store`` exists so a test can supply a double. In production it is
    ``None`` and the PostgreSQL repository is built from the environment.
    """
    resolved = (
        Path(artifact_path) if artifact_path is not None else DEFAULT_ARTIFACT_PATH
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.startup = load_service(resolved, audit_store=audit_store)
        yield
        release_resources(application.state.startup)

    application = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )

    application.state.artifact_path = resolved

    @application.get(HEALTH_ROUTE, response_model=HealthResponse)
    def health(request: Request) -> HealthResponse | JSONResponse:
        """Liveness, artifact identity and audit-store readiness.

        Performs no model inference, and costs one trivial database round trip
        rather than a query against the audit table.
        """
        startup = request.app.state.startup
        database = DATABASE_READY if startup.audit.is_ready() else DATABASE_UNAVAILABLE

        if startup.service is None:
            unavailable = HealthResponse(status="unavailable", database=database)

            return JSONResponse(
                status_code=503,
                content=unavailable.model_dump(),
            )

        service: RiskScoringService = startup.service
        artifact = service.artifact

        return HealthResponse(
            status="ok",
            model_version=artifact.model_version,
            feature_version=artifact.feature_version,
            policy_version=service.policy.config.policy_version,
            artifact=artifact.describe(),
            database=database,
        )

    @application.post(SCORE_ROUTE, response_model=RiskScoreResponse)
    def score(
        payload: RiskScoreRequest,
        service: RiskScoringService = Depends(get_service),
        audit: AuditStore = Depends(get_audit_store),
    ) -> RiskScoreResponse:
        """Score one transaction, explain both halves, then record the decision.

        The write happens after the decision is complete and before the response
        is returned, and it cannot change that decision: nothing on the audit path
        is an input to the model, the calibration or the policy.
        """
        response = to_response(service.score(payload))

        record_decision(audit, payload, response)

        return response

    @application.exception_handler(ArtifactUnavailableError)
    async def artifact_unavailable(
        request: Request,
        error: ArtifactUnavailableError,
    ) -> JSONResponse:
        return _error_response(
            status_code=503,
            code=error.code,
            message=ARTIFACT_UNAVAILABLE_MESSAGE,
        )

    @application.exception_handler(AuditUnavailableError)
    async def audit_store_unavailable(
        request: Request,
        error: AuditUnavailableError,
    ) -> JSONResponse:
        """Report an unaudited decision as a failure, without the decision.

        The cause is logged here, traceback included, and stays here: the response
        carries no SQL, no connection string, no driver message and no path.
        """
        logger.error(
            "decision could not be audited for %s %s",
            request.method,
            request.url.path,
            exc_info=error,
        )

        return _error_response(
            status_code=503,
            code=error.code,
            message=AUDIT_UNAVAILABLE_MESSAGE,
        )

    @application.exception_handler(Exception)
    async def unhandled(request: Request, error: Exception) -> JSONResponse:
        """Report a failure as a failure.

        The traceback is logged server-side and deliberately kept out of the
        response: a client must never receive a filesystem path, a frame of
        library internals, or a plausible-looking probability that was never
        actually computed.
        """
        logger.error(
            "inference failed for %s %s",
            request.method,
            request.url.path,
            exc_info=error,
        )

        return _error_response(
            status_code=500,
            code=INFERENCE_FAILED,
            message=INFERENCE_FAILED_MESSAGE,
        )

    return application


def _error_response(*, status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorOut(code=code, message=message))

    return JSONResponse(status_code=status_code, content=body.model_dump())


app = create_app()
