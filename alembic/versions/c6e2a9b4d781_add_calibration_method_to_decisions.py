"""Persist the calibration method on each risk decision.

Revision ID: c6e2a9b4d781
Revises: b7c4e9d2a610
Create Date: 2026-11-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "c6e2a9b4d781"
down_revision: str | None = "b7c4e9d2a610"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "risk_decisions",
        sa.Column(
            "calibration_method",
            sa.String(length=16),
            server_default="unknown",
            nullable=False,
        ),
    )
    op.create_check_constraint(
        op.f("ck_risk_decisions_calibration_method_is_supported"),
        "risk_decisions",
        "calibration_method IN ('isotonic', 'sigmoid', 'unknown')",
    )
    op.alter_column(
        "risk_decisions",
        "calibration_method",
        server_default=None,
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_risk_decisions_calibration_method_is_supported"),
        "risk_decisions",
        type_="check",
    )
    op.drop_column("risk_decisions", "calibration_method")
