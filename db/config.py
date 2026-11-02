"""Where the audit database is, and nothing else.

The URL comes from the environment so no credential is ever committed. The
default is the local Docker Compose database, whose role and password are
deliberately throwaway: a development convenience, not a production secret.

Only PostgreSQL is accepted. The audit schema stores its explanation payloads as
JSONB and its timestamps as TIMESTAMPTZ, so a different backend would change
what an audit record can hold rather than merely where it is kept.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


DATABASE_URL_ENVIRONMENT_VARIABLE = "DATABASE_URL"

DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://riskforge:riskforge@localhost:5432/riskforge"
)

# The driver RiskForge pins. A bare ``postgresql://`` URL would otherwise make
# SQLAlchemy reach for psycopg2, which is not installed, and fail with an import
# error that says nothing about the URL that caused it.
DRIVER_QUALIFIED_SCHEME = "postgresql+psycopg"
DRIVERLESS_SCHEMES = frozenset({"postgres", "postgresql"})


class DatabaseConfigurationError(RuntimeError):
    """``DATABASE_URL`` is unusable.

    Deliberately carries no URL: a connection string holds a password, and the
    message may reach a log aggregator or an error response.
    """


@dataclass(frozen=True)
class DatabaseConfig:
    """The resolved audit database location."""

    url: str

    @classmethod
    def from_environment(cls) -> DatabaseConfig:
        """Read ``DATABASE_URL``, falling back to the local default."""
        configured = os.environ.get(DATABASE_URL_ENVIRONMENT_VARIABLE, "").strip()
        url = configured or DEFAULT_DATABASE_URL

        return cls(url=_validated(url, was_configured=bool(configured)))


def _validated(url: str, *, was_configured: bool) -> str:
    parts = urlparse(url)
    scheme = parts.scheme.lower()

    if scheme in DRIVERLESS_SCHEMES:
        return url.replace(f"{scheme}://", f"{DRIVER_QUALIFIED_SCHEME}://", 1)

    if scheme != DRIVER_QUALIFIED_SCHEME:
        source = DATABASE_URL_ENVIRONMENT_VARIABLE if was_configured else "default"
        raise DatabaseConfigurationError(
            f"The audit database URL from {source} is not a PostgreSQL URL "
            f"(scheme {scheme!r}). The audit schema requires PostgreSQL JSONB "
            "and TIMESTAMPTZ columns, and RiskForge connects with psycopg 3."
        )

    return url
