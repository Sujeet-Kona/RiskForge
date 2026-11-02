"""Persisting a decision that has already been made.

Nothing here scores, explains or re-derives anything. The audit record is built
from the response the caller is about to receive, so what is stored and what was
reported are the same values by construction rather than by convention.

Writing happens after the decision is complete, and a failed write is never
turned into a successful response. A decision that could not be audited is
reported as a service failure carrying neither its probability nor its decision,
because the stated purpose of the audit store is that a reported decision can be
reconstructed later. Returning the score anyway would claim an audit trail that
does not exist.
"""

from __future__ import annotations

from app.schemas import RiskScoreRequest, RiskScoreResponse
from db.repository import AuditPersistenceError, AuditRecord, AuditStore


AUDIT_UNAVAILABLE = "audit_unavailable"

AUDIT_UNAVAILABLE_MESSAGE = (
    "The risk decision was produced but could not be recorded in the audit "
    "store, so it is not being returned. RiskForge does not report a decision it "
    "cannot audit. Nothing about the decision is included in this response; "
    "retry once the audit store is reachable."
)


class AuditUnavailableError(RuntimeError):
    """A computed decision could not be persisted, so it is not reported."""

    def __init__(self) -> None:
        super().__init__(AUDIT_UNAVAILABLE_MESSAGE)
        self.code = AUDIT_UNAVAILABLE


def build_audit_record(
    request: RiskScoreRequest,
    response: RiskScoreResponse,
) -> AuditRecord:
    """The decision exactly as reported, plus the transaction that produced it.

    ``scored_inputs`` is the scored transaction's own model-contract fields:
    every field of the request payload except the identifier, which is already a
    column. The ``history`` array is deliberately not stored. Persisting it would
    make PostgreSQL a source of behavioral feature history, which Phase 7
    forbids, so the record states how many prior rows were considered instead of
    holding them.
    """
    return AuditRecord(
        transaction_id=response.transaction_id,
        risk_probability=response.risk_probability,
        decision=response.decision,
        reason_codes=[entry.model_dump() for entry in response.reason_codes],
        risk_decreasing_signals=[
            entry.model_dump() for entry in response.risk_decreasing_signals
        ],
        decision_reason_code=response.decision_reason_code.model_dump(),
        decision_explanation=response.decision_explanation,
        model_detail=response.model_detail.model_dump(),
        policy_detail=response.policy_detail.model_dump(),
        scored_inputs=request.transaction.model_dump(exclude={"TransactionID"}),
        top_k=request.top_k,
        model_version=response.model_version,
        feature_version=response.feature_version,
        policy_version=response.policy_version,
        feature_count=response.feature_count,
        history_transactions_used=response.history_transactions_used,
    )


def record_decision(
    store: AuditStore,
    request: RiskScoreRequest,
    response: RiskScoreResponse,
) -> int:
    """Persist the decision ``response`` reports, returning its audit identifier.

    The store is asked once, synchronously. One write per successful inference is
    the whole of Phase 7's persistence budget: there is no queue, no worker and
    no retry, so a failure is terminal for this request.
    """
    try:
        return store.create_audit_record(build_audit_record(request, response))
    except AuditPersistenceError as error:
        raise AuditUnavailableError from error
