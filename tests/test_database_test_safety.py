from __future__ import annotations

import pytest
from sqlalchemy.engine import make_url

from tests.conftest import _audit_test_url


@pytest.mark.parametrize(
    "host",
    ["example.com", "10.10.10.10", "192.168.1.20", "db.internal"],
)
def test_audit_test_database_refuses_non_loopback_hosts(monkeypatch, host):
    monkeypatch.setenv(
        "DATABASE_URL",
        f"postgresql+psycopg://riskforge:secret@{host}:5432/riskforge",
    )

    with pytest.raises(RuntimeError, match="require an explicitly local"):
        _audit_test_url()


@pytest.mark.parametrize("url", ["postgresql+psycopg:///riskforge", "sqlite:///riskforge"])
def test_audit_test_database_refuses_missing_or_non_postgres_hosts(monkeypatch, url):
    monkeypatch.setenv("DATABASE_URL", url)

    with pytest.raises(RuntimeError, match="require an explicitly local"):
        _audit_test_url()


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
def test_audit_test_database_uses_a_dedicated_local_database_and_preserves_query(
    monkeypatch, host
):
    monkeypatch.setenv(
        "DATABASE_URL",
        f"postgresql+psycopg://riskforge:secret@{host}:5432/riskforge?sslmode=disable",
    )

    target = make_url(_audit_test_url())

    assert target.database == "riskforge_test"
    assert target.host == host
    assert target.query == {"sslmode": "disable"}
