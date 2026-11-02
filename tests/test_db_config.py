"""The audit database's configuration, which is the only place a URL is read.

These are the guarantees that keep a credential out of the repository and out of
a client response, and that stop the service from being pointed at a backend
whose column types would silently change what an audit record can hold.
"""

from __future__ import annotations

import pytest

from db.config import (
    DATABASE_URL_ENVIRONMENT_VARIABLE,
    DEFAULT_DATABASE_URL,
    DatabaseConfig,
    DatabaseConfigurationError,
)


def test_the_local_default_is_the_documented_one():
    assert DEFAULT_DATABASE_URL == (
        "postgresql+psycopg://riskforge:riskforge@localhost:5432/riskforge"
    )


def test_the_default_is_used_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv(DATABASE_URL_ENVIRONMENT_VARIABLE, raising=False)

    assert DatabaseConfig.from_environment().url == DEFAULT_DATABASE_URL


def test_a_configured_url_is_used_verbatim(monkeypatch):
    url = "postgresql+psycopg://audit:s3cret@db.internal:6543/riskforge_audit"
    monkeypatch.setenv(DATABASE_URL_ENVIRONMENT_VARIABLE, url)

    assert DatabaseConfig.from_environment().url == url


def test_surrounding_whitespace_is_not_part_of_the_url(monkeypatch):
    monkeypatch.setenv(DATABASE_URL_ENVIRONMENT_VARIABLE, f"  {DEFAULT_DATABASE_URL}\n")

    assert DatabaseConfig.from_environment().url == DEFAULT_DATABASE_URL


def test_a_driverless_url_is_qualified_with_the_pinned_driver(monkeypatch):
    """``postgresql://`` alone would make SQLAlchemy look for psycopg2."""
    monkeypatch.setenv(
        DATABASE_URL_ENVIRONMENT_VARIABLE, "postgresql://riskforge@localhost/riskforge"
    )

    assert DatabaseConfig.from_environment().url == (
        "postgresql+psycopg://riskforge@localhost/riskforge"
    )


@pytest.mark.parametrize(
    "url",
    [
        "sqlite:///audit.db",
        "mysql+pymysql://riskforge@localhost/riskforge",
        "mongodb://localhost/riskforge",
        "not-a-url",
    ],
)
def test_a_non_postgresql_url_is_rejected(monkeypatch, url: str):
    """JSONB and TIMESTAMPTZ are the audit schema's storage, not a preference."""
    monkeypatch.setenv(DATABASE_URL_ENVIRONMENT_VARIABLE, url)

    with pytest.raises(DatabaseConfigurationError):
        DatabaseConfig.from_environment()


def test_the_rejection_names_the_variable_and_not_the_secret(monkeypatch):
    """A rejected URL is reported by variable name; its contents stay private."""
    monkeypatch.setenv(
        DATABASE_URL_ENVIRONMENT_VARIABLE,
        "mysql+pymysql://audit:hunter2@db.internal:3306/riskforge",
    )

    with pytest.raises(DatabaseConfigurationError) as raised:
        DatabaseConfig.from_environment()

    message = str(raised.value)

    assert DATABASE_URL_ENVIRONMENT_VARIABLE in message
    assert "hunter2" not in message
    assert "db.internal" not in message
    assert "3306" not in message


def test_an_empty_value_falls_back_rather_than_failing(monkeypatch):
    monkeypatch.setenv(DATABASE_URL_ENVIRONMENT_VARIABLE, "   ")

    assert DatabaseConfig.from_environment().url == DEFAULT_DATABASE_URL
