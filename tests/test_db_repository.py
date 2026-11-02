"""The repository against a real PostgreSQL database.

Every test here needs a server; the ``audit_database`` fixture skips the module
when none is reachable, and the skip is visible in the run summary. Nothing in
this file talks to a mock, because the thing under test is the round trip: what
the service hands over, what PostgreSQL stores, and what comes back.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from db.config import DatabaseConfig
from db.models import AUDIT_TABLE_NAME, RiskDecision
from db.repository import (
    AUDIT_RECORD_FIELDS,
    AuditPersistenceError,
    AuditRecord,
    AuditRepository,
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
