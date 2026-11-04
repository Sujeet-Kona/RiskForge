"""The MLflow configuration is local, explicit, and carries no secret.

Tracking is offline bookkeeping, so the interesting failures here are not
"cannot connect" but "was pointed somewhere it should not have been": a remote
tracking server, a relative database that moves with the working directory, or a
URI with credentials in it that would then be recorded with every run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tracking.config import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL_NAME,
    DEFAULT_TRACKING_URI,
    FILE_STORE_OPT_OUT_VALUE,
    FILE_STORE_OPT_OUT_VARIABLE,
    MLFLOW_EXPERIMENT_NAME_VARIABLE,
    MLFLOW_REGISTERED_MODEL_NAME_VARIABLE,
    MLFLOW_TRACKING_URI_VARIABLE,
    TrackingConfig,
    TrackingConfigurationError,
    apply_file_store_opt_out,
    resolve_store_path,
)


CONFIGURATION_VARIABLES = (
    MLFLOW_TRACKING_URI_VARIABLE,
    MLFLOW_EXPERIMENT_NAME_VARIABLE,
    MLFLOW_REGISTERED_MODEL_NAME_VARIABLE,
)


@pytest.fixture(autouse=True)
def clear_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from an environment that says nothing about MLflow."""
    for variable in (*CONFIGURATION_VARIABLES, FILE_STORE_OPT_OUT_VARIABLE):
        monkeypatch.delenv(variable, raising=False)


def test_the_defaults_are_a_local_store_and_the_locked_names():
    config = TrackingConfig.from_environment()

    assert config.tracking_uri == DEFAULT_TRACKING_URI == "file:./mlruns"
    assert config.experiment_name == DEFAULT_EXPERIMENT_NAME
    assert config.registered_model_name == DEFAULT_REGISTERED_MODEL_NAME


def test_the_default_store_is_a_file_store():
    assert TrackingConfig.from_environment().is_file_store


@pytest.mark.parametrize(
    ("variable", "value", "attribute"),
    [
        (MLFLOW_TRACKING_URI_VARIABLE, "file:/tmp/elsewhere", "tracking_uri"),
        (MLFLOW_EXPERIMENT_NAME_VARIABLE, "my-experiment", "experiment_name"),
        (MLFLOW_REGISTERED_MODEL_NAME_VARIABLE, "my-model", "registered_model_name"),
    ],
)
def test_each_variable_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, attribute: str
):
    monkeypatch.setenv(variable, value)

    assert getattr(TrackingConfig.from_environment(), attribute) == value


def test_a_blank_registered_model_name_switches_the_registry_off(monkeypatch):
    """Deliberate opt-out, distinct from never having configured one."""
    monkeypatch.setenv(MLFLOW_REGISTERED_MODEL_NAME_VARIABLE, "   ")

    config = TrackingConfig.from_environment()

    assert config.registered_model_name is None


def test_surrounding_whitespace_is_not_part_of_a_configured_value(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_URI_VARIABLE, "  file:/tmp/x  ")
    monkeypatch.setenv(MLFLOW_EXPERIMENT_NAME_VARIABLE, "  spaced  ")

    config = TrackingConfig.from_environment()

    assert config.tracking_uri == "file:/tmp/x"
    assert config.experiment_name == "spaced"


def test_a_bare_directory_is_normalised_to_a_file_uri(monkeypatch):
    """The scheme is recorded so the store location is never ambiguous."""
    monkeypatch.setenv(MLFLOW_TRACKING_URI_VARIABLE, "./somewhere/mlruns")

    config = TrackingConfig.from_environment()

    assert config.tracking_uri == "file:./somewhere/mlruns"
    assert config.is_file_store
    assert config.store_location == "./somewhere/mlruns"


def test_an_absolute_sqlite_store_is_accepted(monkeypatch, tmp_path: Path):
    database = tmp_path / "mlflow.db"
    monkeypatch.setenv(MLFLOW_TRACKING_URI_VARIABLE, f"sqlite:///{database}")

    config = TrackingConfig.from_environment()

    assert config.tracking_uri == f"sqlite:///{database}"
    assert not config.is_file_store
    assert Path(config.store_location) == database


def test_a_relative_sqlite_store_is_rejected(monkeypatch):
    """It would resolve against whichever directory the run was launched from."""
    monkeypatch.setenv(MLFLOW_TRACKING_URI_VARIABLE, "sqlite:///mlflow.db")

    with pytest.raises(TrackingConfigurationError, match="relative SQLite"):
        TrackingConfig.from_environment()


def test_a_sqlite_uri_with_no_database_file_is_rejected(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_URI_VARIABLE, "sqlite:///")

    with pytest.raises(TrackingConfigurationError, match="no database file"):
        TrackingConfig.from_environment()


@pytest.mark.parametrize(
    "uri",
    [
        "http://mlflow.example.com:5000",
        "https://mlflow.example.com",
        "s3://riskforge/mlruns",
        "dbfs:/mnt/mlruns",
        "postgresql://localhost/mlflow",
    ],
)
def test_a_remote_backend_is_rejected(monkeypatch, uri: str):
    """Local-only is a scope decision, and this is where it is enforced."""
    monkeypatch.setenv(MLFLOW_TRACKING_URI_VARIABLE, uri)

    with pytest.raises(TrackingConfigurationError, match="local-only"):
        TrackingConfig.from_environment()


def test_a_uri_that_names_a_host_is_rejected(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_URI_VARIABLE, "file://tracker/mlruns")

    with pytest.raises(TrackingConfigurationError, match="names a host"):
        TrackingConfig.from_environment()


def test_a_uri_carrying_credentials_is_rejected(monkeypatch):
    """A tracking URI is logged and compared in plain text."""
    monkeypatch.setenv(MLFLOW_TRACKING_URI_VARIABLE, "file:./mlruns@hunter2")

    with pytest.raises(TrackingConfigurationError, match="credentials"):
        TrackingConfig.from_environment()


@pytest.mark.parametrize(
    "uri",
    ["file:./mlruns?token=abc", "file:./mlruns#section"],
)
def test_a_uri_with_a_query_or_fragment_is_rejected(monkeypatch, uri: str):
    monkeypatch.setenv(MLFLOW_TRACKING_URI_VARIABLE, uri)

    with pytest.raises(TrackingConfigurationError, match="query or fragment"):
        TrackingConfig.from_environment()


def test_a_file_uri_with_no_directory_is_rejected(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_URI_VARIABLE, "file:")

    with pytest.raises(TrackingConfigurationError, match="does not name a directory"):
        TrackingConfig.from_environment()


@pytest.mark.parametrize(
    "uri",
    ["http://user:hunter2@mlflow.example.com", "file:./mlruns@hunter2"],
)
def test_the_rejection_message_never_echoes_the_uri(monkeypatch, uri: str):
    """The message can reach a log aggregator, so it must not carry the value."""
    monkeypatch.setenv(MLFLOW_TRACKING_URI_VARIABLE, uri)

    with pytest.raises(TrackingConfigurationError) as excinfo:
        TrackingConfig.from_environment()

    message = str(excinfo.value)

    assert "hunter2" not in message
    assert uri not in message


def test_the_file_store_opt_out_is_applied_only_for_a_file_store():
    """MLflow 3 refuses the filesystem backend unless the caller opts in."""
    config = TrackingConfig.from_environment()

    assert apply_file_store_opt_out(config) is True
    assert os.environ[FILE_STORE_OPT_OUT_VARIABLE] == FILE_STORE_OPT_OUT_VALUE


def test_the_file_store_opt_out_leaves_a_database_backend_alone(tmp_path):
    config = TrackingConfig(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name=DEFAULT_EXPERIMENT_NAME,
        registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
    )

    assert apply_file_store_opt_out(config) is False
    assert FILE_STORE_OPT_OUT_VARIABLE not in os.environ


def test_a_relative_store_is_anchored_to_the_repository_root(tmp_path: Path):
    """So the location cannot depend on where the script was launched from."""
    config = TrackingConfig.from_environment()

    resolved = config.resolved_against(tmp_path)

    assert resolved.tracking_uri == f"file:{tmp_path / 'mlruns'}"
    assert resolved.store_location == str(tmp_path / "mlruns")
    assert resolved.experiment_name == config.experiment_name
    assert resolved.registered_model_name == config.registered_model_name


def test_an_absolute_store_is_left_unchanged(tmp_path: Path):
    config = TrackingConfig(
        tracking_uri=f"file:{tmp_path / 'mlruns'}",
        experiment_name=DEFAULT_EXPERIMENT_NAME,
        registered_model_name=None,
    )

    assert config.resolved_against(tmp_path / "elsewhere").tracking_uri == (
        config.tracking_uri
    )


def test_a_database_store_is_left_unchanged_by_anchoring(tmp_path: Path):
    config = TrackingConfig(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name=DEFAULT_EXPERIMENT_NAME,
        registered_model_name=None,
    )

    assert config.resolved_against(tmp_path) is config


def test_resolve_store_path_returns_an_absolute_location(tmp_path: Path):
    path = resolve_store_path(TrackingConfig.from_environment(), root=tmp_path)

    assert path.is_absolute()
    assert path == tmp_path / "mlruns"


def test_the_configuration_is_immutable():
    config = TrackingConfig.from_environment()

    with pytest.raises(AttributeError):
        config.tracking_uri = "file:/tmp/other"  # type: ignore[misc]
