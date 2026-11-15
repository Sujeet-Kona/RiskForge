"""Harden the database boundary for one-way review-case resolution.

Revision ID: d8f4b2c6a901
Revises: c6e2a9b4d781
Create Date: 2026-11-15
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "d8f4b2c6a901"
down_revision: str | None = "c6e2a9b4d781"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION riskforge_guard_review_case_transition()
        RETURNS trigger AS $rf$
        BEGIN
            IF OLD.status <> 'PENDING'
               OR NEW.status NOT IN ('APPROVED', 'REJECTED') THEN
                RAISE EXCEPTION 'RiskForge review cases only resolve once';
            END IF;

            IF NEW.id IS DISTINCT FROM OLD.id
               OR NEW.decision_id IS DISTINCT FROM OLD.decision_id
               OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
                RAISE EXCEPTION 'RiskForge review case identity is immutable';
            END IF;

            IF NEW.updated_at IS NULL
               OR NEW.resolved_at IS NULL
               OR NEW.updated_at IS DISTINCT FROM NEW.resolved_at
               OR NEW.resolved_at < OLD.created_at
               OR NEW.resolved_by IS NULL
               OR btrim(NEW.resolved_by) = '' THEN
                RAISE EXCEPTION 'RiskForge review resolution metadata is invalid';
            END IF;

            RETURN NEW;
        END;
        $rf$ LANGUAGE plpgsql;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION riskforge_guard_review_case_transition()
        RETURNS trigger AS $rf$
        BEGIN
            IF OLD.status <> 'PENDING'
               OR NEW.status NOT IN ('APPROVED', 'REJECTED') THEN
                RAISE EXCEPTION 'RiskForge review cases only resolve once';
            END IF;
            IF NEW.decision_id <> OLD.decision_id
               OR NEW.created_at <> OLD.created_at THEN
                RAISE EXCEPTION 'RiskForge review case identity is immutable';
            END IF;
            RETURN NEW;
        END;
        $rf$ LANGUAGE plpgsql;
        """
    )
