"""Alembic environment for the RiskForge audit store.

The URL comes from :mod:`db.config` and the metadata from :mod:`db.models`, so a
migration is always run against the same database and the same schema the service
uses. There is no second copy of either in this file.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from db.config import DatabaseConfig
from db.models import Base


config = context.config

if config.config_file_name is not None:
    # ``disable_existing_loggers=False`` because a migration can run inside a
    # process that already configured logging — the test suite does — and the
    # default would silence the application's own loggers as a side effect.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Escaped for configparser, which would otherwise read a percent sign in a
# password as the start of an interpolation.
config.set_main_option(
    "sqlalchemy.url", DatabaseConfig.from_environment().url.replace("%", "%%")
)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL without connecting, for review."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations to the configured database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
