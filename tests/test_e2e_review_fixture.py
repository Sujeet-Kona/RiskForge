from unittest.mock import MagicMock

import pytest

from scripts.e2e_review_fixture import database, reset


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://localhost/riskforge",
        "postgresql+psycopg://localhost/riskforge_test",
        "postgresql+psycopg://example.invalid/riskforge_e2e",
        "postgresql+psycopg:///riskforge_e2e",
        "sqlite:///riskforge_e2e",
    ],
)
def test_e2e_fixture_refuses_unsafe_database_before_connecting(monkeypatch, url):
    create = MagicMock()
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr("scripts.e2e_review_fixture.AuditDatabase.create", create)
    with pytest.raises(SystemExit, match="loopback"):
        database()
    create.assert_not_called()


def test_e2e_fixture_accepts_only_explicit_disposable_database(monkeypatch):
    create = MagicMock()
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://localhost/riskforge_e2e")
    monkeypatch.setattr("scripts.e2e_review_fixture.AuditDatabase.create", create)
    assert database() is create.return_value


def test_e2e_reset_refuses_unrelated_decisions():
    db = MagicMock()
    connection = db.engine.begin.return_value.__enter__.return_value
    connection.scalar.return_value = 1
    with pytest.raises(SystemExit, match="non-E2E"):
        reset(db)
    connection.execute.assert_not_called()
