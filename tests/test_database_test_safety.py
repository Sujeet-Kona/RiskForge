from __future__ import annotations

import pytest
from sqlalchemy.engine import make_url

from conftest import _audit_test_url


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


def test_audit_test_database_refuses_postgres_url_without_host(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg:///riskforge")

    with pytest.raises(RuntimeError, match="require an explicitly local"):
        _audit_test_url()


def test_audit_test_database_refuses_non_postgres_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///riskforge")

    with pytest.raises(RuntimeError, match="require a PostgreSQL URL"):
        _audit_test_url()


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
def test_audit_test_database_uses_a_dedicated_local_database_and_preserves_query(
    monkeypatch, host
):
    monkeypatch.setenv(
        "DATABASE_URL",
        f"postgresql+psycopg://riskforge:secret@{'[' + host + ']' if ':' in host else host}:5432/riskforge?sslmode=disable",
    )

    target = make_url(_audit_test_url())

    assert target.database == "riskforge_test"
    assert target.host == host
    assert target.query == {"sslmode": "disable"}



def test_create_test_database_parses_url_and_preserves_query(monkeypatch):

    captured = {}
    executed = []

    class FakeResult:
        def scalar(self):
            return None

    class FakeConnection:
        def execute(self, statement, parameters=None):
            executed.append((str(statement), parameters))
            return FakeResult()

    class FakeEngine:
        def connect(self):
            class ConnectionContext:
                def __enter__(self):
                    return FakeConnection()

                def __exit__(self, *_):
                    return False

            return ConnectionContext()

        def dispose(self):
            captured["disposed"] = True

    def fake_create_engine(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeEngine()

    monkeypatch.setattr("sqlalchemy.create_engine", fake_create_engine)

    from conftest import _create_test_database

    _create_test_database(
        "postgresql+psycopg://riskforge:secret@localhost:5432/riskforge_test?sslmode=disable"
    )

    parsed = make_url(captured["url"])
    assert parsed.database == "postgres"
    assert parsed.query == {"sslmode": "disable"}
    assert captured["kwargs"] == {"isolation_level": "AUTOCOMMIT"}
    assert executed[0][1] == {"name": "riskforge_test"}
    assert executed[1][0] == 'CREATE DATABASE "riskforge_test"'
    assert captured["disposed"] is True

@pytest.mark.parametrize("scheme", ["postgres", "postgresql"])
def test_audit_test_database_normalizes_driverless_postgres_schemes(monkeypatch, scheme):
    monkeypatch.setenv(
        "DATABASE_URL",
        f"{scheme}://riskforge:secret@localhost:5432/riskforge?sslmode=disable",
    )

    target = make_url(_audit_test_url())

    assert target.drivername == "postgresql+psycopg"
    assert target.database == "riskforge_test"
    assert target.query == {"sslmode": "disable"}
