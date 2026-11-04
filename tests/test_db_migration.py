"""The committed migration is the schema, and it is reversible.

These tests need a real PostgreSQL database, because what they prove is a
statement about a server: that running the repository's migration against an
empty database produces exactly the table :mod:`db.models` describes, and that
downgrading takes it away again. A model-only test could not tell the difference
between a migration that was written and a migration that works.

Skipped, visibly, when no PostgreSQL is reachable locally. CI fails instead.
"""

from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.engine.reflection import Inspector

from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

from db.models import AUDIT_TABLE_NAME, RiskDecision


def _inspector(engine: Engine) -> Inspector:
    return inspect(engine)


def _current_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def _head_revision(config) -> str:
    return ScriptDirectory.from_config(config).get_current_head()


def _user_tables(engine: Engine) -> set[str]:
    """Every table that is not Alembic's own bookkeeping."""
    return set(_inspector(engine).get_table_names()) - {"alembic_version"}


# --- there is one revision, and it is the first ------------------------------


def test_the_repository_carries_exactly_one_initial_revision(alembic_config):
    """One linear history: a single base revision and a single head."""
    script = ScriptDirectory.from_config(alembic_config)

    roots = [
        revision
        for revision in script.walk_revisions()
        if revision.down_revision is None
    ]

    assert len(roots) == 1
    assert len(script.get_heads()) == 1


def test_the_database_is_at_head(audit_database, alembic_config):
    assert _current_revision(audit_database.engine) == _head_revision(alembic_config)


def test_the_migration_and_the_models_have_not_drifted_apart(alembic_config):
    """``alembic check``: autogenerate finds nothing left to do.

    This is the assertion that keeps the migration honest. If a column is added
    to :class:`RiskDecision` without a migration, or a migration is written that
    the model does not describe, this fails rather than leaving the repository
    with a schema that only exists on somebody's machine.
    """
    command.check(alembic_config)


# --- what the migration builds ------------------------------------------------


def test_the_migration_creates_the_audit_table(audit_database):
    assert _user_tables(audit_database.engine) == {AUDIT_TABLE_NAME}


def test_the_table_has_exactly_the_modeled_columns(audit_database):
    columns = _inspector(audit_database.engine).get_columns(AUDIT_TABLE_NAME)

    assert {column["name"] for column in columns} == {
        column.name for column in RiskDecision.__table__.columns
    }


def test_the_structured_payloads_are_jsonb_columns(audit_database):
    """JSONB, not JSON: an audit query filters on a reason code."""
    columns = {
        column["name"]: column["type"]
        for column in _inspector(audit_database.engine).get_columns(AUDIT_TABLE_NAME)
    }

    for name in (
        "reason_codes",
        "risk_decreasing_signals",
        "decision_reason_code",
        "model_detail",
        "policy_detail",
        "scored_inputs",
    ):
        assert str(columns[name]) == "JSONB", name


def test_the_timestamp_is_timezone_aware_and_database_generated(audit_database):
    columns = {
        column["name"]: column
        for column in _inspector(audit_database.engine).get_columns(AUDIT_TABLE_NAME)
    }
    created_at = columns["created_at"]

    assert str(created_at["type"]) == "TIMESTAMP"
    assert created_at["type"].timezone is True
    assert created_at["nullable"] is False
    assert created_at["default"] is not None


def test_the_primary_key_is_the_surrogate_id(audit_database):
    assert _inspector(audit_database.engine).get_pk_constraint(AUDIT_TABLE_NAME)[
        "constrained_columns"
    ] == ["id"]


def test_the_migration_creates_the_indexes_it_declares(audit_database):
    reflected = {
        index["name"] for index in _inspector(audit_database.engine).get_indexes(
            AUDIT_TABLE_NAME
        )
    }
    declared = {index.name for index in RiskDecision.__table__.indexes}

    assert declared == {
        "ix_risk_decisions_transaction_id",
        "ix_risk_decisions_created_at",
        "ix_risk_decisions_decision_created_at",
    }
    assert reflected == declared


def test_transaction_id_is_indexed_but_not_unique(audit_database):
    """Scoring the same transaction twice is two decisions, and both are recorded."""
    reflected = _inspector(audit_database.engine).get_indexes(AUDIT_TABLE_NAME)
    unique_constraints = _inspector(audit_database.engine).get_unique_constraints(
        AUDIT_TABLE_NAME
    )

    for index in reflected:
        if index["column_names"] == ["transaction_id"]:
            assert index["unique"] is False

    assert not any(
        constraint["column_names"] == ["transaction_id"]
        for constraint in unique_constraints
    )


def test_the_migration_creates_every_check_constraint(audit_database):
    reflected = {
        constraint["name"]
        for constraint in _inspector(audit_database.engine).get_check_constraints(
            AUDIT_TABLE_NAME
        )
    }

    assert reflected == {
        "ck_risk_decisions_risk_probability_in_unit_interval",
        "ck_risk_decisions_decision_is_a_policy_outcome",
        "ck_risk_decisions_feature_count_is_positive",
        "ck_risk_decisions_history_used_is_not_negative",
        "ck_risk_decisions_top_k_is_positive",
    }


# --- and takes it away again --------------------------------------------------


def test_downgrade_to_base_removes_the_table(audit_database, alembic_config):
    """The migration is reversible, so the schema is reproducible from nothing."""
    try:
        command.downgrade(alembic_config, "base")

        assert _user_tables(audit_database.engine) == set()
        assert _current_revision(audit_database.engine) is None
    finally:
        command.upgrade(alembic_config, "head")


def test_a_full_downgrade_upgrade_cycle_rebuilds_the_same_schema(
    audit_database, alembic_config
):
    before = _inspector(audit_database.engine).get_columns(AUDIT_TABLE_NAME)

    try:
        command.downgrade(alembic_config, "base")
        command.upgrade(alembic_config, "head")

        after = _inspector(audit_database.engine).get_columns(AUDIT_TABLE_NAME)
        assert [(c["name"], str(c["type"])) for c in after] == [
            (c["name"], str(c["type"])) for c in before
        ]
        assert _current_revision(audit_database.engine) == _head_revision(
            alembic_config
        )
    finally:
        if AUDIT_TABLE_NAME not in _user_tables(audit_database.engine):
            command.upgrade(alembic_config, "head")
