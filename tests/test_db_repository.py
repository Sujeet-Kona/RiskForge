"""The repository against a real PostgreSQL database.

Every test here needs a server; the ``audit_database`` fixture skips the module
when none is reachable locally, and the skip is visible in the run summary. CI
fails instead of skipping. Nothing in this file talks to a mock, because the
thing under test is the round trip: what the service hands over, what
PostgreSQL stores, and what comes back.
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from db.config import DatabaseConfig
from db.models import (
    AUDIT_TABLE_NAME,
    RiskDecision,
    RiskReviewEvent,
    RiskIdempotencyRecord,
)
from db.repository import (
    AUDIT_RECORD_FIELDS,
    AuditPersistenceError,
    AuditRecord,
    AuditRepository,
    ReviewConflictError,
    IdempotencyConflictError,
)
from db.session import AuditDatabase


# Port 1 is privileged and nothing listens on it, so a connection is refused
# immediately rather than timing out. This is how "the database is down" is
# reproduced without taking a real one down.
UNREACHABLE_DATABASE_URL = (
    "postgresql+psycopg://riskforge:riskforge@127.0.0.1:1/riskforge"
)


def _unreachable_repository() -> AuditRepository:
    database = AuditDatabase.create(DatabaseConfig(url=UNREACHABLE_DATABASE_URL))

    return AuditRepository(database=database)


def _row_values(record: AuditRecord) -> dict:
    return {name: getattr(record, name) for name in AUDIT_RECORD_FIELDS}


# --- writing and reading back ------------------------------------------------


def test_a_valid_record_is_written_and_read_back_identically(
    audit_repository: AuditRepository, audit_record: AuditRecord
):
    decision_id = audit_repository.create_audit_record(audit_record)

    stored = audit_repository.get_audit_record(decision_id)

    assert stored is not None
    assert dataclasses.replace(stored, id=None, created_at=None) == audit_record


def test_the_write_returns_a_usable_identifier(
    audit_repository: AuditRepository, audit_record: AuditRecord
):
    first = audit_repository.create_audit_record(audit_record)
    second = audit_repository.create_audit_record(audit_record)

    assert isinstance(first, int)
    assert second != first


def test_the_row_really_is_in_the_audit_table(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
    audit_database: AuditDatabase,
):
    """Read with SQL, not with the repository, so nothing is taken on trust."""
    decision_id = audit_repository.create_audit_record(audit_record)

    with audit_database.engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT transaction_id, decision, risk_probability "
                f"FROM {AUDIT_TABLE_NAME} WHERE id = :identifier"
            ),
            {"identifier": decision_id},
        ).one()

    assert row.transaction_id == audit_record.transaction_id
    assert row.decision == "REVIEW"
    assert row.risk_probability == audit_record.risk_probability


def test_the_database_generates_a_timezone_aware_timestamp(
    audit_repository: AuditRepository, audit_record: AuditRecord
):
    before = datetime.now(timezone.utc)

    decision_id = audit_repository.create_audit_record(audit_record)
    stored = audit_repository.get_audit_record(decision_id)

    assert stored.created_at is not None
    assert stored.created_at.tzinfo is not None
    # PostgreSQL renders TIMESTAMPTZ in the session's zone, which is the server's,
    # so the assertion is on the instant rather than on the offset reported.
    assert before <= stored.created_at.astimezone(timezone.utc) <= datetime.now(
        timezone.utc
    )


def test_the_caller_does_not_supply_the_timestamp(
    audit_repository: AuditRepository, audit_record: AuditRecord
):
    """``created_at`` is the database's, so it cannot be backdated by a client."""
    assert audit_record.created_at is None

    decision_id = audit_repository.create_audit_record(audit_record)

    assert audit_repository.get_audit_record(decision_id).created_at is not None


def test_the_probability_survives_at_full_precision(
    audit_repository: AuditRepository, audit_record: AuditRecord
):
    """A rounded store would move a decision across a frozen threshold on paper."""
    probability = 0.09090909090909091
    decision_id = audit_repository.create_audit_record(
        dataclasses.replace(audit_record, risk_probability=probability)
    )

    stored = audit_repository.get_audit_record(decision_id)

    assert stored.risk_probability == probability
    assert stored.risk_probability.hex() == probability.hex()


def test_the_jsonb_payloads_survive_the_round_trip(
    audit_repository: AuditRepository, audit_record: AuditRecord
):
    decision_id = audit_repository.create_audit_record(audit_record)

    stored = audit_repository.get_audit_record(decision_id)

    assert stored.reason_codes == audit_record.reason_codes
    assert stored.risk_decreasing_signals == audit_record.risk_decreasing_signals
    assert stored.decision_reason_code == audit_record.decision_reason_code
    assert stored.model_detail == audit_record.model_detail
    assert stored.policy_detail == audit_record.policy_detail


def test_a_gap_in_the_scored_inputs_stays_a_gap(
    audit_repository: AuditRepository, audit_record: AuditRecord
):
    """JSON null comes back as None, not as the string ``"None"``."""
    decision_id = audit_repository.create_audit_record(audit_record)

    stored = audit_repository.get_audit_record(decision_id)

    assert stored.scored_inputs == audit_record.scored_inputs
    assert stored.scored_inputs["R_emaildomain"] is None


def test_the_same_transaction_identifier_may_be_recorded_twice(
    audit_repository: AuditRepository, audit_record: AuditRecord
):
    """Rescoring an identifier is a second decision, not an update to the first."""
    first = audit_repository.create_audit_record(audit_record)
    second = audit_repository.create_audit_record(
        dataclasses.replace(audit_record, risk_probability=0.5, decision="BLOCK")
    )

    assert first != second
    assert audit_repository.get_audit_record(first).decision == "REVIEW"
    assert audit_repository.get_audit_record(second).decision == "BLOCK"


def test_review_decision_creates_a_pending_case_and_can_be_resolved(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
):
    decision_id = audit_repository.create_audit_record(audit_record)

    cases, total = audit_repository.list_review_cases(
        status="PENDING",
        limit=10,
        offset=0,
    )

    assert total == 1
    case = cases[0]
    assert case.decision_id == decision_id
    assert case.status == "PENDING"
    assert case.decision.decision == "REVIEW"

    resolved = audit_repository.resolve_review_case(
        case_id=case.id,
        outcome="APPROVED",
        reviewer_id="reviewer-01",
        comment="Manual review completed.",
    )

    assert resolved is not None
    assert resolved.status == "APPROVED"
    assert resolved.decision.decision == "REVIEW"
    assert resolved.resolved_by == "reviewer-01"
    assert resolved.resolution_comment == "Manual review completed."

    with audit_repository.database.session_factory() as session:
        event_total = session.query(RiskReviewEvent).filter_by(
            review_case_id=case.id
        ).count()
    assert event_total == 1


def test_a_review_case_cannot_be_resolved_twice(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
):
    audit_repository.create_audit_record(audit_record)
    cases, _ = audit_repository.list_review_cases(
        status="PENDING",
        limit=10,
        offset=0,
    )
    case_id = cases[0].id

    audit_repository.resolve_review_case(
        case_id=case_id,
        outcome="REJECTED",
        reviewer_id="reviewer-01",
        comment=None,
    )

    with pytest.raises(ReviewConflictError):
        audit_repository.resolve_review_case(
            case_id=case_id,
            outcome="APPROVED",
            reviewer_id="reviewer-02",
            comment=None,
        )


def test_review_events_are_database_enforced_append_only(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
):
    audit_repository.create_audit_record(audit_record)
    cases, _ = audit_repository.list_review_cases(
        status="PENDING",
        limit=10,
        offset=0,
    )
    audit_repository.resolve_review_case(
        case_id=cases[0].id,
        outcome="APPROVED",
        reviewer_id="reviewer-01",
        comment=None,
    )

    with pytest.raises(DBAPIError):
        with audit_repository.database.session_factory() as session:
            event = session.query(RiskReviewEvent).filter_by(
                review_case_id=cases[0].id
            ).one()
            event.comment = "rewritten"
            session.commit()


def test_an_identifier_that_was_never_recorded_reads_back_as_none(
    audit_repository: AuditRepository
):
    assert audit_repository.get_audit_record(999_999) is None


def test_a_reachable_database_reports_itself_ready(
    audit_repository: AuditRepository,
):
    assert audit_repository.is_ready() is True


# --- the database's own constraints -----------------------------------------


def test_the_database_rejects_a_probability_python_was_not_asked_about(
    audit_database: AuditDatabase, audit_record: AuditRecord
):
    """The CHECK constraint is not decoration: it fires when Python is bypassed."""
    values = _row_values(audit_record)
    values["risk_probability"] = 1.5

    with pytest.raises(IntegrityError):
        with audit_database.session_factory() as session:
            session.add(RiskDecision(**values))
            session.commit()


def test_the_database_rejects_a_decision_outside_the_policy_vocabulary(
    audit_database: AuditDatabase, audit_record: AuditRecord
):
    values = _row_values(audit_record)
    # Six characters so the value reaches the CHECK constraint instead of being
    # rejected first by the column width, which is what this test is about.
    values["decision"] = "DENIED"

    with pytest.raises(IntegrityError):
        with audit_database.session_factory() as session:
            session.add(RiskDecision(**values))
            session.commit()


def test_a_rejected_row_is_not_left_behind(
    audit_repository: AuditRepository,
    audit_database: AuditDatabase,
    audit_record: AuditRecord,
):
    """A failed write leaves no half-row: the audit store is append-only or clean."""
    values = _row_values(audit_record)
    values["risk_probability"] = -0.5

    with pytest.raises(IntegrityError):
        with audit_database.session_factory() as session:
            session.add(RiskDecision(**values))
            session.commit()

    with audit_database.engine.connect() as connection:
        count = connection.execute(
            text(f"SELECT count(*) FROM {AUDIT_TABLE_NAME}")
        ).scalar_one()

    assert count == 0


# --- a database that is not there -------------------------------------------


def test_a_write_to_an_unreachable_database_fails_as_an_audit_failure(
    audit_record: AuditRecord,
):
    repository = _unreachable_repository()

    try:
        with pytest.raises(AuditPersistenceError):
            repository.create_audit_record(audit_record)
    finally:
        repository.database.dispose()


def test_a_read_from_an_unreachable_database_fails_as_an_audit_failure():
    repository = _unreachable_repository()

    try:
        with pytest.raises(AuditPersistenceError):
            repository.get_audit_record(1)
    finally:
        repository.database.dispose()


def test_an_unreachable_database_reports_itself_not_ready():
    """Readiness is reported, not raised: ``/health`` must be able to answer."""
    repository = _unreachable_repository()

    try:
        assert repository.is_ready() is False
    finally:
        repository.database.dispose()


def test_the_failure_exposes_no_connection_or_driver_detail(audit_record: AuditRecord):
    """The message reaches a client. The cause reaches a log, and stops there."""
    repository = _unreachable_repository()

    try:
        with pytest.raises(AuditPersistenceError) as raised:
            repository.create_audit_record(audit_record)
    finally:
        repository.database.dispose()

    message = str(raised.value)

    for forbidden in (
        "psycopg",
        "postgresql",
        "127.0.0.1",
        "5432",
        "riskforge:riskforge",
        "OperationalError",
        "connection",
        "INSERT",
        "sqlalchemy",
        "Traceback",
    ):
        assert forbidden not in message, forbidden


def test_the_driver_detail_is_kept_as_the_cause(audit_record: AuditRecord):
    """Sanitized for the client, still diagnosable server-side."""
    repository = _unreachable_repository()

    try:
        with pytest.raises(AuditPersistenceError) as raised:
            repository.create_audit_record(audit_record)
    finally:
        repository.database.dispose()

    assert raised.value.__cause__ is not None



def test_original_risk_decisions_are_database_enforced_append_only(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
):
    decision_id = audit_repository.create_audit_record(audit_record)

    with pytest.raises(DBAPIError):
        with audit_repository.database.session_factory() as session:
            decision = session.get(RiskDecision, decision_id)
            decision.risk_probability = 0.5
            session.commit()


def test_a_resolved_review_case_cannot_return_to_pending(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
):
    audit_repository.create_audit_record(audit_record)
    cases, _ = audit_repository.list_review_cases(
        status="PENDING",
        limit=10,
        offset=0,
    )
    case_id = cases[0].id
    audit_repository.resolve_review_case(
        case_id=case_id,
        outcome="APPROVED",
        reviewer_id="reviewer-01",
        comment=None,
    )

    from db.models import RiskReviewCase

    with pytest.raises(DBAPIError):
        with audit_repository.database.session_factory() as session:
            case = session.get(RiskReviewCase, case_id)
            case.status = "PENDING"
            case.resolved_at = None
            case.resolved_by = None
            session.commit()



def test_idempotency_replay_returns_the_original_response_without_a_second_row(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
    audit_database: AuditDatabase,
):
    first = audit_repository.create_idempotent_decision(
        audit_record,
        idempotency_key="riskforge-repo-key-0001",
        request_hash="a" * 64,
        response_body={"decision": "REVIEW", "risk_probability": 0.09},
    )
    replay = audit_repository.create_idempotent_decision(
        audit_record,
        idempotency_key="riskforge-repo-key-0001",
        request_hash="a" * 64,
        response_body={"decision": "REVIEW", "risk_probability": 0.09},
    )

    assert replay == first
    assert audit_repository.get_idempotency_record("riskforge-repo-key-0001") == (
        "a" * 64,
        first,
    )
    with audit_database.engine.connect() as connection:
        count = connection.execute(
            text(f"SELECT COUNT(*) FROM {AUDIT_TABLE_NAME}")
        ).scalar_one()
    assert count == 1


def test_idempotency_key_reuse_with_a_different_hash_conflicts(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
):
    audit_repository.create_idempotent_decision(
        audit_record,
        idempotency_key="riskforge-repo-key-0002",
        request_hash="a" * 64,
        response_body={"decision": "REVIEW"},
    )

    with pytest.raises(IdempotencyConflictError):
        audit_repository.create_idempotent_decision(
            audit_record,
            idempotency_key="riskforge-repo-key-0002",
            request_hash="b" * 64,
            response_body={"decision": "BLOCK"},
        )


def test_idempotency_records_are_database_enforced_append_only(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
):
    audit_repository.create_idempotent_decision(
        audit_record,
        idempotency_key="riskforge-repo-key-0003",
        request_hash="c" * 64,
        response_body={"decision": "REVIEW"},
    )

    with pytest.raises(DBAPIError):
        with audit_repository.database.session_factory() as session:
            stored = session.get(
                RiskIdempotencyRecord,
                "riskforge-repo-key-0003",
            )
            stored.response_body = {"decision": "BLOCK"}
            session.commit()



# --- concurrent idempotency --------------------------------------------------


def test_concurrent_same_key_and_payload_create_one_decision(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
    audit_database: AuditDatabase,
):
    """Simultaneous retries must converge on one committed response and row."""
    worker_count = 8
    start = Barrier(worker_count)
    response_body = {"decision": "REVIEW", "risk_probability": 0.09}

    def submit():
        start.wait(timeout=10)
        return audit_repository.create_idempotent_decision(
            audit_record,
            idempotency_key="riskforge-concurrent-same-0001",
            request_hash="d" * 64,
            response_body=response_body,
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        responses = list(executor.map(lambda _: submit(), range(worker_count)))

    assert responses == [response_body] * worker_count
    assert audit_repository.get_idempotency_record(
        "riskforge-concurrent-same-0001"
    ) == ("d" * 64, response_body)

    with audit_database.engine.connect() as connection:
        decision_count = connection.execute(
            text(f"SELECT COUNT(*) FROM {AUDIT_TABLE_NAME}")
        ).scalar_one()
        review_count = connection.execute(
            text("SELECT COUNT(*) FROM risk_review_cases")
        ).scalar_one()
        idempotency_count = connection.execute(
            text("SELECT COUNT(*) FROM risk_idempotency_records")
        ).scalar_one()

    assert decision_count == 1
    assert review_count == 1
    assert idempotency_count == 1


def test_concurrent_different_payloads_for_one_key_have_one_winner(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
    audit_database: AuditDatabase,
):
    """A racing key reuse cannot commit two decisions or silently change payload."""
    start = Barrier(2)

    def submit(request_hash: str, response_body: dict):
        start.wait(timeout=10)
        try:
            result = audit_repository.create_idempotent_decision(
                audit_record,
                idempotency_key="riskforge-concurrent-conflict-0001",
                request_hash=request_hash,
                response_body=response_body,
            )
        except IdempotencyConflictError:
            return ("conflict", None)
        return ("created", result)

    candidates = [
        ("e" * 64, {"decision": "REVIEW"}),
        ("f" * 64, {"decision": "BLOCK"}),
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(lambda item: submit(*item), candidates)
        )

    assert sorted(outcome for outcome, _ in outcomes) == ["conflict", "created"]
    stored_hash, stored_response = audit_repository.get_idempotency_record(
        "riskforge-concurrent-conflict-0001"
    )
    assert (stored_hash, stored_response) in candidates

    with audit_database.engine.connect() as connection:
        decision_count = connection.execute(
            text(f"SELECT COUNT(*) FROM {AUDIT_TABLE_NAME}")
        ).scalar_one()
        review_count = connection.execute(
            text("SELECT COUNT(*) FROM risk_review_cases")
        ).scalar_one()
        idempotency_count = connection.execute(
            text("SELECT COUNT(*) FROM risk_idempotency_records")
        ).scalar_one()

    assert decision_count == 1
    assert review_count == 1
    assert idempotency_count == 1



def test_review_case_identity_cannot_be_changed_during_resolution(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
):
    audit_repository.create_audit_record(audit_record)
    cases, _ = audit_repository.list_review_cases(
        status="PENDING",
        limit=10,
        offset=0,
    )
    case_id = cases[0].id

    with pytest.raises(DBAPIError):
        with audit_repository.database.session_factory() as session:
            from db.models import RiskReviewCase

            case = session.get(RiskReviewCase, case_id)
            case.id = case_id + 100
            case.status = "APPROVED"
            case.updated_at = datetime.now(timezone.utc)
            case.resolved_at = case.updated_at
            case.resolved_by = "reviewer-01"
            session.commit()


def test_review_case_rejects_inconsistent_resolution_timestamps(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
):
    audit_repository.create_audit_record(audit_record)
    cases, _ = audit_repository.list_review_cases(
        status="PENDING",
        limit=10,
        offset=0,
    )
    case_id = cases[0].id

    with pytest.raises(DBAPIError):
        with audit_repository.database.session_factory() as session:
            from db.models import RiskReviewCase

            case = session.get(RiskReviewCase, case_id)
            invalid_time = case.created_at - timedelta(seconds=1)
            case.status = "APPROVED"
            case.updated_at = invalid_time
            case.resolved_at = invalid_time
            case.resolved_by = "reviewer-01"
            session.commit()


def test_review_case_rejects_blank_reviewer_identity_at_database_boundary(
    audit_repository: AuditRepository,
    audit_record: AuditRecord,
):
    audit_repository.create_audit_record(audit_record)
    cases, _ = audit_repository.list_review_cases(
        status="PENDING",
        limit=10,
        offset=0,
    )
    case_id = cases[0].id

    with pytest.raises(DBAPIError):
        with audit_repository.database.session_factory() as session:
            from db.models import RiskReviewCase

            case = session.get(RiskReviewCase, case_id)
            resolved_at = datetime.now(timezone.utc)
            case.status = "REJECTED"
            case.updated_at = resolved_at
            case.resolved_at = resolved_at
            case.resolved_by = "   "
            session.commit()
