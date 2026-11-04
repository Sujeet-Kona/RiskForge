"""Where MLflow records RiskForge training runs, and nothing else.

Every value comes from the environment so no path, name or credential is ever
committed, and the default is a local filesystem store — this phase introduces
experiment tracking, not infrastructure.

Two properties are enforced here rather than left to the caller:

*Only local backends are accepted.* An ``http(s)://`` tracking server would make
a training run depend on a remote service and would turn the tracking URI into a
secret. MLflow is offline bookkeeping in RiskForge, so a network scheme is a
configuration error, not a deployment option.

*The URI is never reported as configured.* A tracking URI is the one
MLflow setting that can legitimately carry credentials, so the resolved value is
validated to contain none and the dataclass exposes only what is safe to print.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse


MLFLOW_TRACKING_URI_VARIABLE = "MLFLOW_TRACKING_URI"
MLFLOW_EXPERIMENT_NAME_VARIABLE = "MLFLOW_EXPERIMENT_NAME"
MLFLOW_REGISTERED_MODEL_NAME_VARIABLE = "MLFLOW_REGISTERED_MODEL_NAME"

# MLflow 3.x refuses the filesystem backend unless this is set. RiskForge opts
# in explicitly for a URI it has already validated as local, which is the whole
# reason the default below can stay a plain directory.
FILE_STORE_OPT_OUT_VARIABLE = "MLFLOW_ALLOW_FILE_STORE"
FILE_STORE_OPT_OUT_VALUE = "true"

DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_EXPERIMENT_NAME = "riskforge-xgboost-lifecycle"
DEFAULT_REGISTERED_MODEL_NAME = "riskforge-fraud-model"

FILE_SCHEME = "file"
SQLITE_SCHEME = "sqlite"

# A local store is a directory or a SQLite file. Anything that names a host is a
# remote service, and anything that embeds a password is a secret.
LOCAL_SCHEMES = frozenset({FILE_SCHEME, SQLITE_SCHEME})


class TrackingConfigurationError(RuntimeError):
    """The MLflow configuration is unusable.

    Carries no URI: a tracking URI can contain credentials, and this message may
    reach a log aggregator or a terminal that is being shared.
    """


@dataclass(frozen=True)
class TrackingConfig:
    """The resolved location and naming for one tracking backend.

    ``registered_model_name`` is ``None`` when the caller has switched the model
    registry off, in which case a run is still tracked and still carries its
    version metadata as tags — only the registry entry is skipped.
    """

    tracking_uri: str
    experiment_name: str
    registered_model_name: str | None

    @classmethod
    def from_environment(cls) -> TrackingConfig:
        """Read the MLflow variables, falling back to the local defaults."""
        configured_uri = os.environ.get(MLFLOW_TRACKING_URI_VARIABLE, "").strip()
        configured_experiment = os.environ.get(
            MLFLOW_EXPERIMENT_NAME_VARIABLE, ""
        ).strip()
        configured_model = os.environ.get(
            MLFLOW_REGISTERED_MODEL_NAME_VARIABLE, ""
        )

        return cls(
            tracking_uri=_validated_uri(
                configured_uri or DEFAULT_TRACKING_URI,
                was_configured=bool(configured_uri),
            ),
            experiment_name=configured_experiment or DEFAULT_EXPERIMENT_NAME,
            registered_model_name=_registered_model_name(configured_model),
        )

    @property
    def is_file_store(self) -> bool:
        """True when the tracking backend is a directory rather than a database."""
        return urlparse(self.tracking_uri).scheme == FILE_SCHEME

    @property
    def store_location(self) -> str:
        """The filesystem location holding run data, without the URI scheme.

        Both accepted backends name one: a ``file:`` store is a directory and a
        ``sqlite:///`` store is a single database file. Reported instead of the
        raw URI so that a log line, a test assertion and a ``.gitignore`` entry
        all refer to the same thing.
        """
        return _store_location(self.tracking_uri)

    def resolved_against(self, root: Path) -> TrackingConfig:
        """A copy whose store location does not depend on the working directory.

        A relative ``file:`` URI is where MLflow would otherwise create
        ``mlruns/`` under whatever directory the process was launched from, which
        is how generated run data ends up somewhere nobody gitignored. SQLite
        URIs are already required to be absolute, so they pass through unchanged.
        """
        if not self.is_file_store:
            return self

        location = resolve_store_path(self, root=root)

        return replace(self, tracking_uri=f"{FILE_SCHEME}:{location}")


def _registered_model_name(configured: str) -> str | None:
    """Blank means the registry is deliberately switched off."""
    if MLFLOW_REGISTERED_MODEL_NAME_VARIABLE not in os.environ:
        return DEFAULT_REGISTERED_MODEL_NAME

    name = configured.strip()

    return name or None


def _validated_uri(uri: str, *, was_configured: bool) -> str:
    parts = urlparse(uri)
    scheme = parts.scheme.lower()
    source = MLFLOW_TRACKING_URI_VARIABLE if was_configured else "default"

    if scheme and scheme not in LOCAL_SCHEMES:
        raise TrackingConfigurationError(
            f"The MLflow tracking URI from {source} uses scheme {scheme!r}. "
            "RiskForge tracking is local-only: use a directory "
            f"({FILE_SCHEME}:./mlruns) or a SQLite file "
            f"({SQLITE_SCHEME}:///path/to/mlflow.db)."
        )

    if parts.netloc:
        raise TrackingConfigurationError(
            f"The MLflow tracking URI from {source} names a host. A local store "
            "has no host, and a remote one would make training depend on a "
            "service this repository does not run."
        )

    if "@" in uri:
        raise TrackingConfigurationError(
            f"The MLflow tracking URI from {source} contains credentials. "
            "Remove them: a tracking URI is logged and compared in plain text."
        )

    if parts.query or parts.fragment:
        raise TrackingConfigurationError(
            f"The MLflow tracking URI from {source} carries a query or "
            "fragment. Tokens passed that way would be recorded with every run."
        )

    if scheme == SQLITE_SCHEME:
        database = _store_location(uri)

        if not database.strip():
            raise TrackingConfigurationError(
                f"The MLflow tracking URI from {source} is a SQLite scheme with "
                "no database file."
            )

        # A relative SQLite path would be resolved against whichever directory
        # the training script happened to be launched from, so the same
        # configuration would silently produce a different database each time.
        if not Path(database).is_absolute():
            raise TrackingConfigurationError(
                f"The MLflow tracking URI from {source} points at a relative "
                "SQLite database. Use an absolute path (four slashes: "
                f"{SQLITE_SCHEME}:////absolute/path/mlflow.db) so the store "
                "location does not depend on the working directory."
            )

        return uri

    location = parts.path

    if not location.strip():
        raise TrackingConfigurationError(
            f"The MLflow tracking URI from {source} does not name a directory."
        )

    return f"{FILE_SCHEME}:{location}"


def _store_location(uri: str) -> str:
    """The filesystem path a tracking URI points at, for either backend.

    ``sqlite:`` follows the SQLAlchemy convention that MLflow inherits: three
    slashes mean a relative database and four mean an absolute one, so exactly
    one leading slash is part of the URI syntax rather than the path.
    """
    parts = urlparse(uri)

    if parts.scheme == SQLITE_SCHEME and parts.path.startswith("/"):
        return parts.path[1:]

    return parts.path


def apply_file_store_opt_out(config: TrackingConfig) -> bool:
    """Permit the filesystem backend when the configuration asks for one.

    MLflow 3.x raises for ``file:`` stores unless the caller opts out, so the
    opt-out is set here — immediately before the store is opened, and only for a
    URI this module has already validated as local. Returns whether it was
    needed, so a run can say so honestly instead of silently relying on it.

    A database backend is left untouched: setting the variable for one would be
    a change to the process environment with no purpose.
    """
    if not config.is_file_store:
        return False

    os.environ[FILE_STORE_OPT_OUT_VARIABLE] = FILE_STORE_OPT_OUT_VALUE

    return True


def resolve_store_path(config: TrackingConfig, *, root: Path) -> Path:
    """The absolute directory or database file a run's data will land in.

    A relative ``file:`` URI is resolved against ``root`` rather than the
    process working directory, so the same configuration produces the same
    location no matter which directory the training script was launched from.
    """
    location = Path(config.store_location)

    if not location.is_absolute():
        location = root / location

    return location.resolve()
