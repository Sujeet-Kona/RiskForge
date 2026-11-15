"""What the API does about persistence, and what it refuses to do.

Two halves, deliberately kept apart:

The first half runs against the in-memory audit double. It proves the contract
between the endpoint and the store — that what is recorded is what was reported,
that a decision which cannot be recorded is not reported at all, and that the
failure response says nothing about a database.

The second half runs against PostgreSQL. It proves the same thing happened for
real: a 200 leaves a row, the row holds the response's values, and a driver that
cannot reach its server produces the same sanitized 503 rather than a 500 with a
connection string in it.
"""

from __future__ import annotations

import dataclasses
import json

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.audit import AUDIT_UNAVAILABLE, AUDIT_UNAVAILABLE_MESSAGE
from app.config import DEFAULT_TOP_K
from app.main import AUDIT_ROUTE, SCORE_ROUTE, create_app
from app.schemas import RiskScoreResponse
from db.config import DatabaseConfig
from db.models import AUDIT_TABLE_NAME
from db.repository import AuditPersistenceError, AuditRecord, AuditRepository
from db.session import AuditDatabase


UNREACHABLE_DATABASE_URL = (
    "postgresql+psycopg://riskforge:riskforge@127.0.0.1:1/riskforge"
)

REPORTED_FIELDS = frozenset(RiskScoreResponse.model_fields)

FORBIDDEN_IN_A_FAILURE_RESPONSE = (
    "psycopg",
    "postgresql",
    "sqlalchemy",
    "alembic",
    "DATABASE_URL",
    "127.0.0.1",
    "localhost",
    "5432",
    "riskforge:riskforge",
    "INSERT",
    "SELECT",
    "Traceback",
    "site-packages",
    ".py",
)


def _score(client: TestClient, payload: dict) -> dict:
    response = client.post(SCORE_ROUTE, json=payload)

    assert response.status_code == 200, response.text

    return response.json()


# --- the record is the response ------------------------------------------------


def test_a_successful_score_is_recorded(client, audit_store, score_payload):
    body = _score(client, score_payload)

    assert len(audit_store.records) == 1
    assert audit_store.records[0].transaction_id == body["transaction_id"]


def test_every_reported_value_is_the_value_that_was_recorded(
    client, audit_store, score_payload
):
    """Stored == reported, so a reconstructed decision cannot disagree with one
    that was already sent to a caller."""
    body = _score(client, score_payload)
    record = audit_store.records[0]

    assert record.risk_probability == body["risk_probability"]
    assert record.decision == body["decision"]
    assert record.reason_codes == body["reason_codes"]
    assert record.risk_decreasing_signals == body["risk_decreasing_signals"]
    assert record.decision_reason_code == body["decision_reason_code"]
    assert record.decision_explanation == body["decision_explanation"]
    assert record.model_detail == body["model_detail"]
    assert record.policy_detail == body["policy_detail"]
    assert record.model_version == body["model_version"]
    assert record.feature_version == body["feature_version"]
    assert record.calibration_method == body["calibration_method"]
    assert record.policy_version == body["policy_version"]
    assert record.feature_count == body["feature_count"]
    assert record.history_transactions_used == body["history_transactions_used"]


def test_the_recorded_inputs_are_the_transaction_that_was_scored(
    client, audit_store, score_payload
):
    """Everything except the identifier, which is recorded in its own column."""
    body = _score(client, score_payload)
    scored_inputs = audit_store.records[0].scored_inputs

    assert "TransactionID" not in scored_inputs
    assert scored_inputs == {
        key: value
        for key, value in score_payload["transaction"].items()
        if key != "TransactionID"
    }
    assert audit_store.records[0].transaction_id == body["transaction_id"]


def test_the_supplied_history_is_not_persisted(client, audit_store, score_payload):
    """The audit store records decisions; it must never become a history source.

    If prior transactions were written here, a future phase could read them back
    as behavioral context and the point-in-time contract would be enforced by
    convention instead of by the data that exists.
    """
    _score(client, score_payload)
    record = audit_store.records[0]

    assert "history" not in {field.name for field in dataclasses.fields(record)}

    serialized = json.dumps(dataclasses.asdict(record), default=str)
    for entry in score_payload["history"]:
        assert str(entry["TransactionDT"]) not in serialized
        assert str(entry["TransactionAmt"]) not in serialized


def test_the_requested_top_k_is_recorded(client, audit_store, score_payload):
    """``top_k`` changes the explanation, so an audit needs it to reproduce one."""
    payload = {**score_payload, "top_k": 1}

    _score(client, payload)

    assert audit_store.records[0].top_k == 1
    assert len(audit_store.records[0].reason_codes) == 1


def test_the_default_top_k_is_recorded_too(client, audit_store, score_payload):
    _score(client, score_payload)

    assert audit_store.records[0].top_k == DEFAULT_TOP_K


# --- the response contract did not change -------------------------------------


def test_persistence_added_no_field_to_the_response(client, score_payload):
    """Phase 6 callers keep working: no new field, no renamed one."""
    body = _score(client, score_payload)

    assert set(body) == set(REPORTED_FIELDS)


def test_nothing_is_recorded_for_a_request_that_was_rejected(
    client, audit_store, score_payload
):
    """A 422 produced no decision, so there is nothing to audit."""
    not_prior = {
        **score_payload["history"][0],
        "TransactionDT": score_payload["transaction"]["TransactionDT"],
    }
    payload = {**score_payload, "history": [not_prior]}

    response = client.post(SCORE_ROUTE, json=payload)

    assert response.status_code == 422
    assert audit_store.records == []


# --- read-only decision history -----------------------------------------------


def test_decision_history_returns_newest_first_page_and_total(
    client, audit_store, score_payload
):
    first = _score(client, score_payload)
    second_payload = {
        **score_payload,
        "transaction": {
            **score_payload["transaction"],
            "TransactionID": "RF-HISTORY-SECOND",
        },
    }
    second = _score(client, second_payload)

    response = client.get(AUDIT_ROUTE, params={"limit": 1, "offset": 0})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["limit"] == 1
    assert body["offset"] == 0
    assert len(body["items"]) == 1
    assert body["items"][0]["transaction_id"] == second["transaction_id"]
    assert body["items"][0]["decision"] == second["decision"]
    assert body["items"][0]["risk_probability"] == second["risk_probability"]
    assert body["items"][0]["model_version"] == "xgb-001"
    assert body["items"][0]["feature_version"] == "behavioral-v1"
    assert body["items"][0]["calibration_method"] == "isotonic"
    assert body["items"][0]["policy_version"] == "policy-001"
    assert first["transaction_id"] != second["transaction_id"]


def test_decision_history_filters_by_policy_outcome(client, score_payload):
    scored = _score(client, score_payload)

    response = client.get(
        AUDIT_ROUTE,
        params={"decision": scored["decision"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["decision"] == scored["decision"]


def test_decision_history_rejects_unknown_outcome(client):
    response = client.get(AUDIT_ROUTE, params={"decision": "DECLINE"})

    assert response.status_code == 422


def test_decision_history_bounds_page_size(client):
    assert client.get(AUDIT_ROUTE, params={"limit": 101}).status_code == 422
    assert client.get(AUDIT_ROUTE, params={"limit": 0}).status_code == 422
    assert client.get(AUDIT_ROUTE, params={"offset": -1}).status_code == 422


# --- a decision that cannot be audited is not returned ------------------------


def test_an_unrecordable_decision_is_refused_rather_than_returned(
    client, audit_store, score_payload
):
    """The point of the audit store is auditability.

    Returning the score anyway would tell the caller a decision was recorded when
    it was not — the one false claim this phase must never make.
    """
    audit_store.failure = AuditPersistenceError("the audit store is unreachable")

    response = client.post(SCORE_ROUTE, json=score_payload)

    assert response.status_code == 503
    assert response.json() == {
        "error": {"code": AUDIT_UNAVAILABLE, "message": AUDIT_UNAVAILABLE_MESSAGE}
    }


def test_the_refusal_carries_no_part_of_the_decision(
    client, audit_store, score_payload
):
    body = client.post(SCORE_ROUTE, json=score_payload).json()
    audit_store.failure = AuditPersistenceError("the audit store is unreachable")

    refused = client.post(SCORE_ROUTE, json=score_payload).json()

    assert refused["error"]["code"] == AUDIT_UNAVAILABLE
    assert set(refused["error"]) == {"code", "message"}
    assert str(body["risk_probability"]) not in json.dumps(refused)
    assert body["decision"] not in json.dumps(refused)


def test_the_refusal_says_nothing_about_the_database(
    client, audit_store, score_payload
):
    """Sanitized: no driver, no DSN, no SQL, no path, no traceback."""
    audit_store.failure = AuditPersistenceError(
        "psycopg.OperationalError connection to postgresql://riskforge@localhost"
    )

    text = client.post(SCORE_ROUTE, json=score_payload).text

    for fragment in FORBIDDEN_IN_A_FAILURE_RESPONSE:
        assert fragment not in text, fragment


def test_the_failure_is_logged_with_its_cause(
    client, audit_store, score_payload, caplog
):
    """The client gets a sanitized error; the operator gets the real one."""
    audit_store.failure = AuditPersistenceError("the audit store is unreachable")

    with caplog.at_level("ERROR"):
        assert client.post(SCORE_ROUTE, json=score_payload).status_code == 503

    logged = [record for record in caplog.records if record.levelname == "ERROR"]

    assert logged
    assert any(record.exc_info is not None for record in logged)


def test_a_recovered_audit_store_scores_again(client, audit_store, score_payload):
    """The failure is per-request: nothing is latched, nothing needs a restart."""
    audit_store.failure = AuditPersistenceError("the audit store is unreachable")
    assert client.post(SCORE_ROUTE, json=score_payload).status_code == 503

    audit_store.failure = None

    assert client.post(SCORE_ROUTE, json=score_payload).status_code == 200
    assert len(audit_store.records) == 1


# --- idempotent scoring retries ----------------------------------------------


def test_same_idempotency_key_and_payload_replay_the_original_response(
    client, audit_store, score_payload
):
    headers = {"Idempotency-Key": "riskforge-request-0001"}
    first = client.post(SCORE_ROUTE, json=score_payload, headers=headers)
    second = client.post(SCORE_ROUTE, json=score_payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    assert len(audit_store.records) == 1
    assert len(audit_store.idempotency_records) == 1


def test_reusing_an_idempotency_key_for_a_different_payload_conflicts(
    client, audit_store, score_payload
):
    headers = {"Idempotency-Key": "riskforge-request-0002"}
    first = client.post(SCORE_ROUTE, json=score_payload, headers=headers)
    changed = {
        **score_payload,
        "transaction": {
            **score_payload["transaction"],
            "TransactionAmt": score_payload["transaction"]["TransactionAmt"] + 1.0,
        },
    }

    conflict = client.post(SCORE_ROUTE, json=changed, headers=headers)

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert len(audit_store.records) == 1


def test_malformed_idempotency_key_is_rejected(client, score_payload):
    response = client.post(
        SCORE_ROUTE,
        json=score_payload,
        headers={"Idempotency-Key": "bad key"},
    )

    assert response.status_code == 422


# --- human review is separate from the automated decision --------------------


def test_only_review_decisions_create_review_cases(audit_store, audit_record):
    audit_store.create_audit_record(audit_record)
    assert len(audit_store.review_cases) == 1
    assert audit_store.review_cases[0].status == "PENDING"
    assert audit_store.review_cases[0].decision.decision == "REVIEW"

    allowed = dataclasses.replace(audit_record, decision="ALLOW")
    audit_store.create_audit_record(allowed)
    assert len(audit_store.review_cases) == 1


def test_review_queue_requires_configured_reviewer_credentials(
    client, audit_store, audit_record, monkeypatch
):
    audit_store.create_audit_record(audit_record)
    monkeypatch.delenv("RISKFORGE_REVIEWER_API_KEY", raising=False)
    monkeypatch.delenv("RISKFORGE_REVIEWER_ID", raising=False)

    response = client.get("/v1/reviews")

    assert response.status_code == 503


def test_review_queue_rejects_an_invalid_reviewer_key(
    client, audit_store, audit_record, monkeypatch
):
    audit_store.create_audit_record(audit_record)
    monkeypatch.setenv("RISKFORGE_REVIEWER_API_KEY", "correct-key")
    monkeypatch.setenv("RISKFORGE_REVIEWER_ID", "reviewer-01")

    response = client.get(
        "/v1/reviews",
        headers={"X-Reviewer-API-Key": "wrong-key"},
    )

    assert response.status_code == 401


def test_authenticated_review_queue_returns_original_decision(
    client, audit_store, audit_record, monkeypatch
):
    audit_store.create_audit_record(audit_record)
    monkeypatch.setenv("RISKFORGE_REVIEWER_API_KEY", "test-review-key")
    monkeypatch.setenv("RISKFORGE_REVIEWER_ID", "reviewer-01")

    response = client.get(
        "/v1/reviews",
        headers={"X-Reviewer-API-Key": "test-review-key"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    case = body["items"][0]
    assert case["status"] == "PENDING"
    assert case["decision"] == "REVIEW"
    assert case["transaction_id"] == audit_record.transaction_id
    assert case["model_version"] == audit_record.model_version
    assert case["calibration_method"] == audit_record.calibration_method
    assert case["reason_codes"] == audit_record.reason_codes


def test_reviewer_resolution_is_recorded_without_rewriting_model_decision(
    client, audit_store, audit_record, monkeypatch
):
    audit_store.create_audit_record(audit_record)
    monkeypatch.setenv("RISKFORGE_REVIEWER_API_KEY", "test-review-key")
    monkeypatch.setenv("RISKFORGE_REVIEWER_ID", "reviewer-01")
    headers = {"X-Reviewer-API-Key": "test-review-key"}

    response = client.post(
        "/v1/reviews/1/resolve",
        headers=headers,
        json={"outcome": "APPROVED", "comment": "Reviewed supporting context."},
    )

    assert response.status_code == 200
    case = response.json()["case"]
    assert case["status"] == "APPROVED"
    assert case["decision"] == "REVIEW"
    assert case["resolved_by"] == "reviewer-01"
    assert case["resolution_comment"] == "Reviewed supporting context."
    assert len(audit_store.review_events) == 1
    assert audit_store.review_events[0]["action"] == "APPROVED"

    duplicate = client.post(
        "/v1/reviews/1/resolve",
        headers=headers,
        json={"outcome": "REJECTED"},
    )
    assert duplicate.status_code == 409


def test_review_resolution_rejects_missing_cases(client, monkeypatch):
    monkeypatch.setenv("RISKFORGE_REVIEWER_API_KEY", "test-review-key")
    monkeypatch.setenv("RISKFORGE_REVIEWER_ID", "reviewer-01")

    response = client.post(
        "/v1/reviews/999/resolve",
        headers={"X-Reviewer-API-Key": "test-review-key"},
        json={"outcome": "REJECTED"},
    )

    assert response.status_code == 404


# --- readiness is reported without changing liveness --------------------------


def test_health_reports_a_ready_audit_store(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["calibration_method"] == "isotonic"
    assert body["database"] == "ready"


def test_health_reports_an_unreachable_audit_store_as_still_alive(client, audit_store):
    """/health keeps its Phase 6 meaning: it reports the artifact.

    A service whose audit store is down cannot return a decision, and readiness
    says so, but it is alive and its model is loaded — so it stays a 200 with
    ``status`` unchanged rather than becoming a second failure signal.
    """
    audit_store.ready = False

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["database"] == "unavailable"


# --- the same, against PostgreSQL --------------------------------------------


def test_a_score_leaves_one_row_in_the_database(
    persisted_client, audit_repository, score_payload
):
    body = _score(persisted_client, score_payload)

    with audit_repository.database.engine.connect() as connection:
        rows = connection.execute(
            text(f"SELECT id, decision FROM {AUDIT_TABLE_NAME}")
        ).all()

    assert len(rows) == 1
    assert rows[0].decision == body["decision"]

    stored = audit_repository.get_audit_record(rows[0].id)
    assert isinstance(stored, AuditRecord)
    assert stored.transaction_id == body["transaction_id"]
    assert stored.risk_probability == body["risk_probability"]
    assert stored.reason_codes == body["reason_codes"]
    assert stored.decision_reason_code == body["decision_reason_code"]
    assert stored.model_detail == body["model_detail"]
    assert stored.scored_inputs == {
        key: value
        for key, value in score_payload["transaction"].items()
        if key != "TransactionID"
    }
    assert stored.created_at is not None


def test_scoring_the_same_transaction_twice_records_both_decisions(
    persisted_client, audit_repository, score_payload
):
    _score(persisted_client, score_payload)
    _score(persisted_client, score_payload)

    with audit_repository.database.engine.connect() as connection:
        count = connection.execute(
            text(f"SELECT COUNT(*) FROM {AUDIT_TABLE_NAME}")
        ).scalar_one()

    assert count == 2


def test_postgres_idempotency_replays_without_a_second_decision(
    persisted_client, audit_repository, score_payload
):
    headers = {"Idempotency-Key": "riskforge-postgres-0001"}
    first = _score_with_headers(persisted_client, score_payload, headers)
    second_response = persisted_client.post(SCORE_ROUTE, json=score_payload, headers=headers)

    assert second_response.status_code == 200
    assert second_response.json() == first
    with audit_repository.database.engine.connect() as connection:
        decision_count = connection.execute(
            text(f"SELECT COUNT(*) FROM {AUDIT_TABLE_NAME}")
        ).scalar_one()
        idempotency_count = connection.execute(
            text("SELECT COUNT(*) FROM risk_idempotency_records")
        ).scalar_one()
    assert decision_count == 1
    assert idempotency_count == 1


def _score_with_headers(client: TestClient, payload: dict, headers: dict) -> dict:
    response = client.post(SCORE_ROUTE, json=payload, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def test_decision_history_reads_persisted_postgres_rows(
    persisted_client, score_payload
):
    scored = _score(persisted_client, score_payload)

    response = persisted_client.get(AUDIT_ROUTE, params={"decision": scored["decision"]})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["id"] is not None
    assert item["transaction_id"] == scored["transaction_id"]
    assert item["created_at"] is not None


def test_the_database_reports_itself_ready_through_health(persisted_client):
    body = persisted_client.get("/health").json()

    assert body["status"] == "ok"
    assert body["database"] == "ready"


def test_a_real_driver_failure_produces_the_same_sanitized_503(
    tiny_artifact_path, score_payload
):
    """Not a double: psycopg failing to reach a server, end to end."""
    database = AuditDatabase.create(DatabaseConfig(url=UNREACHABLE_DATABASE_URL))

    try:
        with TestClient(
            create_app(
                artifact_path=tiny_artifact_path,
                audit_store=AuditRepository(database=database),
            )
        ) as test_client:
            response = test_client.post(SCORE_ROUTE, json=score_payload)
    finally:
        database.dispose()

    assert response.status_code == 503
    assert response.json() == {
        "error": {"code": AUDIT_UNAVAILABLE, "message": AUDIT_UNAVAILABLE_MESSAGE}
    }
    for fragment in FORBIDDEN_IN_A_FAILURE_RESPONSE:
        assert fragment not in response.text, fragment


def test_health_reports_an_unreachable_database_as_still_alive(
    tiny_artifact_path, score_payload
):
    database = AuditDatabase.create(DatabaseConfig(url=UNREACHABLE_DATABASE_URL))

    try:
        with TestClient(
            create_app(
                artifact_path=tiny_artifact_path,
                audit_store=AuditRepository(database=database),
            )
        ) as test_client:
            response = test_client.get("/health")
            assert test_client.post(SCORE_ROUTE, json=score_payload).status_code == 503
    finally:
        database.dispose()

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["database"] == "unavailable"
