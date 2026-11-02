"""The RiskForge Phase 6 inference application.

Two endpoints, one scoring path:

``GET /health``
    Liveness plus the identity of the loaded artifact. Performs no inference.

``POST /v1/risk/score``
    Scores one transaction against caller-supplied strictly-prior history and
    returns the calibrated probability, the frozen policy decision, the Phase 5
    reason codes and the version identifiers of everything that produced them.

The service is stateless. Phase 6 is stateless: persistent transaction history
is introduced in Phase 7, so a caller that wants behavioral features must send
the prior transactions those features are computed from.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import DEFAULT_ARTIFACT_PATH, STATELESS_NOTE
from app.dependencies import (
    ARTIFACT_UNAVAILABLE_MESSAGE,
    ArtifactUnavailableError,
    get_service,
    load_service,
)
from app.schemas import (
    ErrorOut,
    ErrorResponse,
    HealthResponse,
    RiskScoreRequest,
    RiskScoreResponse,
)
from app.service import RiskScoringService, to_response


logger = logging.getLogger("riskforge.serving")

API_TITLE = "RiskForge Inference API"
API_VERSION = "phase-6"

HEALTH_ROUTE = "/health"
SCORE_ROUTE = "/v1/risk/score"

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

{STATELESS_NOTE} Behavioral features are computed from the `history` array the
caller supplies; the service stores nothing between requests and connects to no
database.

The model, calibration and preprocessing are loaded once at startup from a frozen
artifact. This service never trains, never fits a calibrator and never falls back
to a substitute model: if the artifact cannot be loaded, scoring returns 503.
"""


def create_app(*, artifact_path: Path | None = None) -> FastAPI:
    """Build the application, loading its artifacts once during startup."""
    resolved = (
        Path(artifact_path) if artifact_path is not None else DEFAULT_ARTIFACT_PATH
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.startup = load_service(resolved)
        yield

    application = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )

    application.state.artifact_path = resolved

    @application.get(HEALTH_ROUTE, response_model=HealthResponse)
    def health(request: Request) -> HealthResponse | JSONResponse:
        """Liveness and artifact identity. Performs no model inference."""
        startup = request.app.state.startup

        if startup.service is None:
            unavailable = HealthResponse(status="unavailable")

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
        )

    @application.post(SCORE_ROUTE, response_model=RiskScoreResponse)
    def score(
        payload: RiskScoreRequest,
        service: RiskScoringService = Depends(get_service),
    ) -> RiskScoreResponse:
        """Score one transaction and explain both halves of the result."""
        return to_response(service.score(payload))

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
