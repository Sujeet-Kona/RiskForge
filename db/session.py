"""The engine and session factory for the audit database.

Sessions are created per operation and closed by the caller's context manager;
the service holds no session between requests, so the audit store adds no state
to a process that is otherwise stateless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from db.config import DatabaseConfig


logger = logging.getLogger("riskforge.audit")


@dataclass(frozen=True)
class AuditDatabase:
    """One engine, and the sessions it produces.

    SQLAlchemy creates engines lazily: constructing this object opens no
    connection, so an unreachable database is discovered when a decision is
    written rather than when the service starts.
    """

    engine: Engine
    session_factory: sessionmaker[Session]

    @classmethod
    def create(cls, config: DatabaseConfig) -> AuditDatabase:
        engine = create_engine(config.url, pool_pre_ping=True)

        return cls(
            engine=engine,
            session_factory=sessionmaker(bind=engine, expire_on_commit=False),
        )

    def is_ready(self) -> bool:
        """Whether the audit store answers a trivial query right now.

        One round trip, no table access: this exists so ``/health`` can report
        readiness without costing a real query.
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError as error:
            logger.error("audit database is not reachable: %s", error)
            return False

        return True

    def dispose(self) -> None:
        """Release the connection pool. Called once, at application shutdown."""
        self.engine.dispose()
