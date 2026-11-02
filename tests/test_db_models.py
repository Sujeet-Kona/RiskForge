"""The audit schema and its boundary type, checked without a database.

Two things are asserted here. First, that the table holds what a reconstruction
needs, in types that hold it faithfully: JSONB for the explanation payloads, a
timezone-aware timestamp, a double-precision probability. Second, that the
``AuditRecord`` the serving layer fills in and the table it lands in cannot drift
apart, and that the record rejects a malformed decision in Python rather than
leaving the whole job to a database constraint.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from sqlalchemy import CheckConstraint, DateTime, Double
from sqlalchemy.dialects.postgresql import JSONB

from db.models import AUDIT_TABLE_NAME, DECISION_VALUES, RiskDecision
from db.repository import AUDIT_RECORD_FIELDS, AuditRecord
from policy.decision import Decision


TABLE = RiskDecision.__table__
COLUMNS = set(TABLE.columns.keys())

# The minimum the Phase 7 brief requires, so that a decision can be
# reconstructed: what was scored, the probability, the decision, the model, the
# feature contract, the policy, the reason codes, the explanation, and when.
REQUIRED_AUDIT_FIELDS = {
    "transaction_id",
    "risk_probability",
    "decision",
    "reason_codes",
    "risk_decreasing_signals",
    "decision_reason_code",
    "decision_explanation",
    "model_version",
    "feature_version",
    "policy_version",
    "feature_count",
    "history_transactions_used",
    "created_at",
}

GENERATED_COLUMNS = {"id", "created_at"}

JSONB_COLUMNS = {
    "reason_codes",
    "risk_decreasing_signals",
    "decision_reason_code",
    "model_detail",
    "policy_detail",
    "scored_inputs",
}


# --- the table ---------------------------------------------------------------


def test_the_audit_table_is_named_after_what_a_row_is():
    assert AUDIT_TABLE_NAME == "risk_decisions"
    assert TABLE.name == AUDIT_TABLE_NAME


def test_every_required_audit_field_is_a_column():
    assert REQUIRED_AUDIT_FIELDS <= COLUMNS


def test_the_table_has_a_surrogate_primary_key():
    assert [column.name for column in TABLE.primary_key] == ["id"]


def test_the_explanation_payloads_are_jsonb():
    for name in JSONB_COLUMNS:
        assert isinstance(TABLE.columns[name].type, JSONB), name


def test_the_probability_is_double_precision():
    assert isinstance(TABLE.columns["risk_probability"].type, Double)


def test_the_timestamp_is_timezone_aware():
    column_type = TABLE.columns["created_at"].type

    assert isinstance(column_type, DateTime)
    assert column_type.timezone is True


def test_the_timestamp_is_set_by_the_database():
    assert TABLE.columns["created_at"].server_default is not None


def test_no_audit_column_is_optional():
    """A reconstruction with a hole in it is worse than no reconstruction."""
    optional = [name for name in COLUMNS if TABLE.columns[name].nullable]

    assert optional == []


def test_the_decision_vocabulary_comes_from_the_policy_enum():
    """ALLOW/REVIEW/BLOCK are stated once, in ``policy.decision``."""
    assert DECISION_VALUES == tuple(member.value for member in Decision)
    assert DECISION_VALUES == ("ALLOW", "REVIEW", "BLOCK")

    constraint = _check_constraint("decision_is_a_policy_outcome")

    for value in DECISION_VALUES:
        assert f"'{value}'" in str(constraint.sqltext)

    assert TABLE.columns["decision"].type.length == max(
        len(value) for value in DECISION_VALUES
    )


@pytest.mark.parametrize(
    "constraint_name,expected",
    [
        ("risk_probability_in_unit_interval", "risk_probability"),
        ("decision_is_a_policy_outcome", "decision"),
        ("feature_count_is_positive", "feature_count"),
        ("history_used_is_not_negative", "history_transactions_used"),
        ("top_k_is_positive", "top_k"),
    ],
)
def test_the_database_enforces_the_ranges_itself(constraint_name: str, expected: str):
    """Python validates first, but the database does not trust that it did."""
    constraint = _check_constraint(constraint_name)

    assert expected in str(constraint.sqltext)


def test_the_indexes_are_the_three_the_audit_queries_need():
    indexes = {
        index.name: [column.name for column in index.columns]
        for index in TABLE.indexes
    }

    assert indexes == {
        "ix_risk_decisions_transaction_id": ["transaction_id"],
        "ix_risk_decisions_created_at": ["created_at"],
        "ix_risk_decisions_decision_created_at": ["decision", "created_at"],
    }


def test_no_index_claims_a_transaction_identifier_is_unique():
    """The identifier is caller-supplied, so scoring it twice is two decisions."""
    for index in TABLE.indexes:
        assert index.unique is not True, index.name


# --- the boundary type ------------------------------------------------------


def test_the_record_fields_map_onto_columns_one_to_one():
    """A field added to the record and not to the table would be dropped."""
    assert set(AUDIT_RECORD_FIELDS) <= COLUMNS
    assert COLUMNS - GENERATED_COLUMNS == set(AUDIT_RECORD_FIELDS)


def test_a_complete_record_is_accepted(audit_record: AuditRecord):
    assert audit_record.transaction_id == "3000001"


@pytest.mark.parametrize("decision", ["ALLOW", "REVIEW", "BLOCK"])
def test_each_policy_outcome_is_accepted(audit_record: AuditRecord, decision: str):
    assert dataclasses.replace(audit_record, decision=decision).decision == decision


@pytest.mark.parametrize("decision", ["allow", "DECLINE", "", "FRAUD"])
def test_a_decision_outside_the_policy_vocabulary_is_rejected(
    audit_record: AuditRecord, decision: str
):
    with pytest.raises(ValueError, match="decision"):
        dataclasses.replace(audit_record, decision=decision)


@pytest.mark.parametrize("probability", [0.0, 1.0, 0.075, 0.110, 0.5])
def test_a_probability_inside_the_unit_interval_is_accepted(
    audit_record: AuditRecord, probability: float
):
    replaced = dataclasses.replace(audit_record, risk_probability=probability)

    assert replaced.risk_probability == probability


@pytest.mark.parametrize(
    "probability", [-0.001, 1.0001, math.nan, math.inf, -math.inf]
)
def test_a_probability_outside_the_unit_interval_is_rejected(
    audit_record: AuditRecord, probability: float
):
    with pytest.raises(ValueError, match="risk_probability"):
        dataclasses.replace(audit_record, risk_probability=probability)


@pytest.mark.parametrize("probability", ["0.5", True, None])
def test_a_probability_that_is_not_a_real_number_is_rejected(
    audit_record: AuditRecord, probability: float
):
    with pytest.raises(TypeError, match="risk_probability"):
        dataclasses.replace(audit_record, risk_probability=probability)


@pytest.mark.parametrize(
    "field_name", ["model_version", "feature_version", "policy_version"]
)
@pytest.mark.parametrize("value", ["", "   ", None])
def test_a_missing_version_identifier_is_rejected(
    audit_record: AuditRecord, field_name: str, value
):
    """Without the versions, a stored decision cannot be attributed to a model."""
    with pytest.raises((ValueError, TypeError)):
        dataclasses.replace(audit_record, **{field_name: value})


@pytest.mark.parametrize("field_name", ["transaction_id", "decision_explanation"])
def test_a_blank_required_text_field_is_rejected(
    audit_record: AuditRecord, field_name: str
):
    with pytest.raises(ValueError):
        dataclasses.replace(audit_record, **{field_name: "  "})


@pytest.mark.parametrize(
    "field_name",
    [
        "decision_reason_code",
        "model_detail",
        "policy_detail",
        "scored_inputs",
    ],
)
def test_a_structured_payload_that_is_not_a_mapping_is_rejected(
    audit_record: AuditRecord, field_name: str
):
    with pytest.raises(TypeError):
        dataclasses.replace(audit_record, **{field_name: ["not", "a", "mapping"]})


@pytest.mark.parametrize("field_name", ["reason_codes", "risk_decreasing_signals"])
def test_a_reason_code_list_must_hold_mappings(
    audit_record: AuditRecord, field_name: str
):
    with pytest.raises(TypeError):
        dataclasses.replace(audit_record, **{field_name: [{"code": "RC"}, "nope"]})


@pytest.mark.parametrize(
    "field_name,value,minimum",
    [
        ("feature_count", 0, 1),
        ("feature_count", -1, 1),
        ("history_transactions_used", -1, 0),
        ("top_k", 0, 1),
    ],
)
def test_a_count_below_its_minimum_is_rejected(
    audit_record: AuditRecord, field_name: str, value: int, minimum: int
):
    with pytest.raises(ValueError, match=field_name):
        dataclasses.replace(audit_record, **{field_name: value})


def test_history_may_legitimately_be_empty(audit_record: AuditRecord):
    """A first-ever transaction has no prior rows, and that is not an error."""
    replaced = dataclasses.replace(audit_record, history_transactions_used=0)

    assert replaced.history_transactions_used == 0


def test_a_record_is_immutable(audit_record: AuditRecord):
    with pytest.raises(dataclasses.FrozenInstanceError):
        audit_record.risk_probability = 1.0


def _check_constraint(short_name: str) -> CheckConstraint:
    name = f"ck_{AUDIT_TABLE_NAME}_{short_name}"

    for constraint in TABLE.constraints:
        if isinstance(constraint, CheckConstraint) and constraint.name == name:
            return constraint

    raise AssertionError(f"no check constraint named {name}")
