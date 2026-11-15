"""Add a separate, append-only human-review workflow.

Revision ID: 9a1f7c3d5e20
Revises: 7eccd8f63a8c
Create Date: 2026-11-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "9a1f7c3d5e20"
down_revision: str | None = "7eccd8f63a8c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "risk_review_cases",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("decision_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=8),
            server_default="PENDING",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.String(length=64), nullable=True),
        sa.Column("resolution_comment", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('PENDING', 'APPROVED', 'REJECTED')",
            name=op.f("ck_risk_review_cases_status_is_a_review_state"),
        ),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["risk_decisions.id"],
            name=op.f("fk_risk_review_cases_decision_id_risk_decisions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_risk_review_cases")),
        sa.UniqueConstraint(
            "decision_id",
            name=op.f("uq_risk_review_cases_decision_id"),
        ),
    )
    op.create_index(
        "ix_risk_review_cases_status_created_at",
        "risk_review_cases",
        ["status", "created_at"],
        unique=False,
    )

    op.create_table(
        "risk_review_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("review_case_id", sa.BigInteger(), nullable=False),
        sa.Column("action", sa.String(length=8), nullable=False),
        sa.Column("reviewer_id", sa.String(length=64), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "action IN ('APPROVED', 'REJECTED')",
            name=op.f("ck_risk_review_events_action_is_a_review_resolution"),
        ),
        sa.ForeignKeyConstraint(
            ["review_case_id"],
            ["risk_review_cases.id"],
            name=op.f("fk_risk_review_events_review_case_id_risk_review_cases"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_risk_review_events")),
    )
    op.create_index(
        "ix_risk_review_events_case_created_at",
        "risk_review_events",
        ["review_case_id", "created_at"],
        unique=False,
    )

    # A decision and a review event are historical facts. Protect both at the
    # database boundary; only the separate review-case state may be updated.
    op.execute(
        """
        CREATE FUNCTION riskforge_reject_audit_mutation()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'RiskForge audit records are append-only';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_risk_decisions_append_only
        BEFORE UPDATE OR DELETE ON risk_decisions
        FOR EACH ROW EXECUTE FUNCTION riskforge_reject_audit_mutation();
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_risk_review_events_append_only
        BEFORE UPDATE OR DELETE ON risk_review_events
        FOR EACH ROW EXECUTE FUNCTION riskforge_reject_audit_mutation();
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_risk_review_events_append_only ON risk_review_events"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_risk_decisions_append_only ON risk_decisions"
    )
    op.execute("DROP FUNCTION IF EXISTS riskforge_reject_audit_mutation()")
    op.drop_index(
        "ix_risk_review_events_case_created_at",
        table_name="risk_review_events",
    )
    op.drop_table("risk_review_events")
    op.drop_index(
        "ix_risk_review_cases_status_created_at",
        table_name="risk_review_cases",
    )
    op.drop_table("risk_review_cases")
