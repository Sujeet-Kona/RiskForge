"""The audit store's schema: one append-only table of decisions already made.

A row is written after a decision has been produced and explained, and is never
updated. It exists so a decision can be reconstructed later — which probability,
which band, under which model, feature contract and policy version, with which
reason codes — not so the model can be retrained or re-scored from it.

Two deliberate omissions. The table holds no label, because the audit store is
not a training set. And it holds no behavioral history, because Phase 7 must not
turn PostgreSQL into a feature source: the 13 behavioral features stay governed
by the Phase 6 serving contract, computed from the history a caller supplies with
the request. ``scored_inputs`` therefore records the scored transaction's own
model-contract fields and ``history_transactions_used`` records how many prior
rows were considered, which is enough to know what was scored without being
enough to score it again from the database alone.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Double,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from policy.decision import Decision


AUDIT_TABLE_NAME = "risk_decisions"

# The decision vocabulary has one home, policy.decision.Decision. Restating
# ALLOW/REVIEW/BLOCK here would be a second place for it to drift, so the CHECK
# constraint and the column width are both derived from the enum.
DECISION_VALUES: tuple[str, ...] = tuple(member.value for member in Decision)

_DECISION_VALUES_SQL = ", ".join(f"'{value}'" for value in DECISION_VALUES)
_DECISION_COLUMN_WIDTH = max(len(value) for value in DECISION_VALUES)

# Explicit names, so a migration never emits a constraint whose name depends on
# the order SQLAlchemy happened to create things in.
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """The audit store's declarative base. Nothing outside ``db`` inherits."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class RiskDecision(Base):
    """One risk decision, as the API reported it, plus when it was recorded.

    ``transaction_id`` is indexed but not unique. It is supplied by the caller,
    so scoring the same identifier twice legitimately produces two audit rows:
    they are two decisions, and collapsing them would lose one.
    """

    __tablename__ = AUDIT_TABLE_NAME
    __table_args__ = (
        CheckConstraint(
            "risk_probability >= 0.0 AND risk_probability <= 1.0",
            name="risk_probability_in_unit_interval",
        ),
        CheckConstraint(
            f"decision IN ({_DECISION_VALUES_SQL})",
            name="decision_is_a_policy_outcome",
        ),
        CheckConstraint("feature_count > 0", name="feature_count_is_positive"),
        CheckConstraint(
            "history_transactions_used >= 0",
            name="history_used_is_not_negative",
        ),
        CheckConstraint("top_k >= 1", name="top_k_is_positive"),
        Index("ix_risk_decisions_transaction_id", "transaction_id"),
        Index("ix_risk_decisions_created_at", "created_at"),
        Index("ix_risk_decisions_decision_created_at", "decision", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    transaction_id: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_probability: Mapped[float] = mapped_column(Double, nullable=False)
    decision: Mapped[str] = mapped_column(String(_DECISION_COLUMN_WIDTH), nullable=False)

    reason_codes: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    risk_decreasing_signals: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False
    )
    decision_reason_code: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    decision_explanation: Mapped[str] = mapped_column(Text, nullable=False)
    model_detail: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    policy_detail: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    scored_inputs: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(32), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False)
    history_transactions_used: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
