"""Add idempotent scoring request records.

Revision ID: b7c4e9d2a610
Revises: 9a1f7c3d5e20
Create Date: 2026-11-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op


revision: str = "b7c4e9d2a610"
down_revision: str | None = "9a1f7c3d5e20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "risk_idempotency_records",
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("decision_id", sa.BigInteger(), nullable=False),
        sa.Column("response_body", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(request_hash) = 64",
            name=op.f("ck_risk_idempotency_records_request_hash_is_sha256"),
        ),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["risk_decisions.id"],
            name=op.f("fk_risk_idempotency_records_decision_id_risk_decisions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "idempotency_key",
            name=op.f("pk_risk_idempotency_records"),
        ),
        sa.UniqueConstraint(
            "decision_id",
            name=op.f("uq_risk_idempotency_records_decision_id"),
        ),
    )
    op.create_index(
        "ix_risk_idempotency_records_created_at",
        "risk_idempotency_records",
        ["created_at"],
        unique=False,
    )
    op.execute(
        """
        CREATE TRIGGER trg_risk_idempotency_records_append_only
        BEFORE UPDATE OR DELETE ON risk_idempotency_records
        FOR EACH ROW EXECUTE FUNCTION riskforge_reject_audit_mutation();
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_risk_idempotency_records_append_only "
        "ON risk_idempotency_records"
    )
    op.drop_index(
        "ix_risk_idempotency_records_created_at",
        table_name="risk_idempotency_records",
    )
    op.drop_table("risk_idempotency_records")
