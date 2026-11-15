"""The RiskForge inference application.

Scoring, audit-history, and review endpoints over one scoring path:

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

import hashlib
import json
import logging
import re
from typing import Literal
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.audit import (
    AUDIT_UNAVAILABLE_MESSAGE,
    AuditUnavailableError,
    list_decisions,
    list_review_cases,
    get_idempotency_response,
    record_idempotent_decision,
    record_decision,
    resolve_review_case,
)
from app.config import DEFAULT_ARTIFACT_PATH, HISTORY_NOTE
from app.dependencies import (
    ARTIFACT_UNAVAILABLE_MESSAGE,
    ArtifactUnavailableError,
    get_audit_store,
    get_service,
    load_service,
    release_resources,
    require_reviewer,
)
from app.schemas import (
    AuditDecisionListResponse,
    AuditDecisionSummaryOut,
    ErrorOut,
    ErrorResponse,
    HealthResponse,
    ReviewCaseOut,
    ReviewQueueResponse,
    ReviewResolutionRequest,
    ReviewResolutionResponse,
    RiskScoreRequest,
    RiskScoreResponse,
)
from app.service import RiskScoringService, to_response
from db.repository import (
    AuditStore,
    IdempotencyConflictError,
    ReviewCaseNotFoundError,
    ReviewCaseRecord,
    ReviewConflictError,
)
from policy.decision import Decision


logger = logging.getLogger("riskforge.serving")

API_TITLE = "RiskForge Inference API"
API_VERSION = "phase-7"

HEALTH_ROUTE = "/health"
SCORE_ROUTE = "/v1/risk/score"
AUDIT_ROUTE = "/v1/audit/decisions"

DATABASE_READY = "ready"
DATABASE_UNAVAILABLE = "unavailable"

INFERENCE_FAILED = "inference_failed"
INFERENCE_FAILED_MESSAGE = (
    "Scoring failed for this transaction. No risk probability and no decision "
    "are being returned; a failure is never reported as a score."
)

UNAUTHENTICATED = "unauthenticated"
NOT_FOUND = "not_found"
METHOD_NOT_ALLOWED = "method_not_allowed"
CONFLICT = "conflict"
REVIEWER_UNCONFIGURED = "reviewer_unconfigured"
FALLBACK_ERROR_CODE = "request_failed"

REVIEWER_CREDENTIALS_REQUIRED_MESSAGE = (
    "Valid reviewer credentials are required."
)
RESOURCE_NOT_FOUND_MESSAGE = "The requested resource was not found."
METHOD_NOT_ALLOWED_MESSAGE = (
    "This method is not allowed on the requested resource."
)
CONFLICT_MESSAGE = (
    "The request conflicts with a decision or review case that has already "
    "been recorded."
)
REVIEWER_UNCONFIGURED_MESSAGE = (
    "Human-review operations are disabled until reviewer credentials are "
    "configured."
)
REQUEST_FAILED_MESSAGE = "The request could not be completed."

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
            calibration_method=artifact.calibration_method,
            policy_version=service.policy.config.policy_version,
            artifact=artifact.describe(),
            database=database,
        )

    @application.post(SCORE_ROUTE, response_model=RiskScoreResponse)
    def score(
        payload: RiskScoreRequest,
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
            min_length=8,
            max_length=128,
        ),
        service: RiskScoringService = Depends(get_service),
        audit: AuditStore = Depends(get_audit_store),
    ) -> RiskScoreResponse:
        """Score one transaction, explain both halves, then record the decision.

        The write happens after the decision is complete and before the response
        is returned, and it cannot change that decision: nothing on the audit path
        is an input to the model, the calibration or the policy.
        """
        request_hash: str | None = None
        if idempotency_key is not None:
            if re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", idempotency_key) is None:
                raise HTTPException(
                    status_code=422,
                    detail="Idempotency-Key may contain only letters, digits, '.', '_', ':', and '-'.",
                )
            canonical_request = json.dumps(
                payload.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            request_hash = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
            try:
                cached = get_idempotency_response(
                    audit,
                    idempotency_key=idempotency_key,
                    request_hash=request_hash,
                )
            except IdempotencyConflictError as error:
                raise HTTPException(
                    status_code=409,
                    detail="Idempotency-Key was already used for a different request.",
                ) from error
            if cached is not None:
                return RiskScoreResponse.model_validate(cached)

        response = to_response(service.score(payload))

        if idempotency_key is not None and request_hash is not None:
            try:
                stored = record_idempotent_decision(
                    audit,
                    payload,
                    response,
                    idempotency_key=idempotency_key,
                    request_hash=request_hash,
                )
            except IdempotencyConflictError as error:
                raise HTTPException(
                    status_code=409,
                    detail="Idempotency-Key was already used for a different request.",
                ) from error
            return RiskScoreResponse.model_validate(stored)

        record_decision(audit, payload, response)

        return response

    @application.get(AUDIT_ROUTE, response_model=AuditDecisionListResponse)
    def decision_history(
        limit: int = Query(default=25, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        decision: Decision | None = None,
        audit: AuditStore = Depends(get_audit_store),
    ) -> AuditDecisionListResponse:
        """Return a bounded, newest-first page of recorded decisions."""
        records, total = list_decisions(
            audit,
            limit=limit,
            offset=offset,
            decision=None if decision is None else decision.value,
        )
        return AuditDecisionListResponse(
            items=[
                AuditDecisionSummaryOut(
                    id=record.id,
                    transaction_id=record.transaction_id,
                    risk_probability=record.risk_probability,
                    decision=record.decision,
                    model_version=record.model_version,
                    feature_version=record.feature_version,
                    calibration_method=record.calibration_method,
                    policy_version=record.policy_version,
                    created_at=record.created_at,
                )
                for record in records
            ],
            total=total,
            limit=limit,
            offset=offset,
        )

    @application.get("/v1/reviews", response_model=ReviewQueueResponse)
    def review_queue(
        status: Literal["PENDING", "APPROVED", "REJECTED"] | None = Query(
            default="PENDING"
        ),
        limit: int = Query(default=25, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        reviewer_id: str = Depends(require_reviewer),
        audit: AuditStore = Depends(get_audit_store),
    ) -> ReviewQueueResponse:
        """Return the authenticated review queue and original model decisions."""
        cases, total = list_review_cases(
            audit,
            status=status,
            limit=limit,
            offset=offset,
        )
        return ReviewQueueResponse(
            items=[_review_case_out(case) for case in cases],
            total=total,
            limit=limit,
            offset=offset,
        )

    @application.post(
        "/v1/reviews/{case_id}/resolve",
        response_model=ReviewResolutionResponse,
    )
    def resolve_review(
        case_id: int,
        payload: ReviewResolutionRequest,
        reviewer_id: str = Depends(require_reviewer),
        audit: AuditStore = Depends(get_audit_store),
    ) -> ReviewResolutionResponse:
        """Resolve one pending review without changing its automated decision."""
        try:
            case = resolve_review_case(
                audit,
                case_id=case_id,
                outcome=payload.outcome,
                reviewer_id=reviewer_id,
                comment=payload.comment,
            )
        except ReviewCaseNotFoundError as error:
            raise HTTPException(status_code=404, detail="Review case not found.") from error
        except ReviewConflictError as error:
            raise HTTPException(
                status_code=409,
                detail="This review case has already been resolved.",
            ) from error

        return ReviewResolutionResponse(case=_review_case_out(case))

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

    @application.exception_handler(StarletteHTTPException)
    async def http_error(
        request: Request,
        error: StarletteHTTPException,
    ) -> JSONResponse:
        """Render an ``HTTPException`` in the same envelope as every other failure.

        Registered on the Starlette base class so the framework's own 404 and 405
        are covered too, not only the raises in this module.

        Each branch names its message constant literally rather than reading
        ``error.detail``, because
        ``test_every_client_facing_error_message_is_a_module_level_constant``
        requires that no runtime-composed string reaches a caller. The status is
        what selects the message, so a 409 from the score route and a 409 from
        the review route share one wording; the request path already tells the
        caller which resource conflicted.

        422 is deliberately left in FastAPI's own ``{"detail": [...]}`` shape by
        delegating to the framework handler. Request-validation failures carry
        the schema's point-in-time rejection reason, and two tests require that
        reason to reach the caller — see ``docs/system-design.md``.
        """
        status = error.status_code

        if status == 422:
            return await http_exception_handler(request, error)

        if status == 401:
            return _error_response(
                status_code=status,
                code=UNAUTHENTICATED,
                message=REVIEWER_CREDENTIALS_REQUIRED_MESSAGE,
            )

        if status == 404:
            return _error_response(
                status_code=status,
                code=NOT_FOUND,
                message=RESOURCE_NOT_FOUND_MESSAGE,
            )

        if status == 405:
            return _error_response(
                status_code=status,
                code=METHOD_NOT_ALLOWED,
                message=METHOD_NOT_ALLOWED_MESSAGE,
            )

        if status == 409:
            return _error_response(
                status_code=status,
                code=CONFLICT,
                message=CONFLICT_MESSAGE,
            )

        if status == 503:
            return _error_response(
                status_code=status,
                code=REVIEWER_UNCONFIGURED,
                message=REVIEWER_UNCONFIGURED_MESSAGE,
            )

        return _error_response(
            status_code=status,
            code=FALLBACK_ERROR_CODE,
            message=REQUEST_FAILED_MESSAGE,
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


def _review_case_out(case: ReviewCaseRecord) -> ReviewCaseOut:
    """Render review state alongside the original immutable scoring record."""
    decision = case.decision
    return ReviewCaseOut(
        case_id=case.id,
        decision_id=case.decision_id,
        status=case.status,
        created_at=case.created_at,
        updated_at=case.updated_at,
        resolved_at=case.resolved_at,
        resolved_by=case.resolved_by,
        resolution_comment=case.resolution_comment,
        transaction_id=decision.transaction_id,
        risk_probability=decision.risk_probability,
        decision=decision.decision,
        reason_codes=decision.reason_codes,
        risk_decreasing_signals=decision.risk_decreasing_signals,
        decision_reason_code=decision.decision_reason_code,
        decision_explanation=decision.decision_explanation,
        model_detail=decision.model_detail,
        policy_detail=decision.policy_detail,
        scored_inputs=decision.scored_inputs,
        model_version=decision.model_version,
        feature_version=decision.feature_version,
        calibration_method=decision.calibration_method,
        policy_version=decision.policy_version,
    )


def _error_response(*, status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorOut(code=code, message=message))

    return JSONResponse(status_code=status_code, content=body.model_dump())


app = create_app()
