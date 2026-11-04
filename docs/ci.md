# RiskForge CI

GitHub Actions validates that the committed code, Alembic schema, and
automated tests still work together. It is a quality gate, not a training
pipeline and not a production deployment.

This document describes `.github/workflows/ci.yml`. It does not claim that a
GitHub Actions run has passed unless a run has been observed.

---

## 1. Workflow overview

The workflow is named **RiskForge CI**. It runs on:

- every push to `main`
- every pull request targeting `main`

The job runs on `ubuntu-latest` with Python 3.12. PostgreSQL 17 is provided as
a GitHub Actions service container, with:

```text
POSTGRES_DB=riskforge
POSTGRES_USER=riskforge
POSTGRES_PASSWORD=riskforge
```

Port `5432:5432` is published. The service health check is `pg_isready` against
that database and role. The job sets:

```text
DATABASE_URL=postgresql+psycopg://riskforge:riskforge@localhost:5432/riskforge
```

Those values match the local development defaults already recorded in
`.env.example`. They are CI service credentials, not production secrets.

---

## 2. Dependency installation

Dependencies are installed from the pinned `requirements.txt`. Ruff is installed
separately at `0.17.3`, the same version used locally. It is a CI and developer
tool, not a runtime dependency, so it is not added to `requirements.txt`.

Pinned application and test packages are not changed by this pipeline.

---

## 3. Quality checks

Two static checks run before any database work:

1. **Ruff** — `python -m ruff check .`
2. **compileall** — `python -m compileall -q .`

Ruff is configured in `ruff.toml`. The enforced set is deliberately narrow:
pyflakes (`F`) plus the pycodestyle families that catch broken or dead code
(`E4`, `E7`, `E9`) — unused imports, unused locals, and f-strings with no
placeholder. Those are defects. The check does not auto-fix or reformat files.

The broader families Ruff can check — import sorting (`I`), modernisation
(`UP`), simplification (`SIM`), bugbear (`B`), and implicit string
concatenation (`ISC`) — are left unenforced. Clearing them would mean rewriting
files this project has no reason to touch. A recent count of those unenforced
families on this tree:

| Code | Count | Rule |
| --- | ---: | --- |
| I001 | 72 | unsorted-imports |
| B905 | 6 | zip-without-explicit-strict |
| UP017 | 4 | datetime-timezone-utc |
| SIM117 | 3 | multiple-with-statements |
| SIM300 | 3 | yoda-conditions |
| B008 | 2 | function-call-in-default-argument |
| SIM114 | 2 | if-with-same-arms |
| ISC004 | 2 | implicit-string-concatenation-in-collection-literal |
| B017 | 1 | assert-raises-exception |
| UP012 | 1 | unnecessary-encode-utf8 |
| **total** | **96** | |

That is not a claim that the repository is lint-clean outside the enforced
rules. It is the reason those families stay off.

`compileall` is a syntax and bytecode check over the tree. It does not execute
tests or import the serving artifact.

---

## 4. Database checks

Against the fresh CI `riskforge` database the workflow runs:

```bash
alembic upgrade head
alembic check
alembic current
```

Then it asserts that the current revision is exactly `7eccd8f63a8c`, the
committed migration head. An unexpected revision fails the job. The migration
schema is not edited to make CI pass.

Alembic resolves the URL through `db.config.DatabaseConfig`, the same path the
service uses. `alembic.ini` does not contain a URL.

The pytest PostgreSQL fixture does **not** reuse the `riskforge` database. It
derives a separate database named `riskforge_test`, creates it if it is
missing, downgrades to Alembic `base`, and upgrades to `head`. That is why the
CI role must be allowed to create databases. The official `postgres:17` image
initialized with `POSTGRES_USER=riskforge` grants that privilege to the role.

---

## 5. Test suite

The last step is:

```bash
python -m pytest -q
```

The full suite runs. Tests that need a model use synthetic fixtures assembled
through the real 28-feature contract; they do not load IEEE-CIS data or the
production serving artifact. Tests that need PostgreSQL use the real
`riskforge_test` database described above.

Locally, if PostgreSQL is unreachable, the `audit_database` fixture skips and
the skip is reported. That is convenient for unit work. It is not acceptable in
CI: a skipped integration suite would look like a green run.

CI therefore sets `RISKFORGE_REQUIRE_POSTGRES=1` (GitHub Actions also sets
`CI=true`). The same fixture then fails instead of skipping. Unrelated skips
are not banned; only the PostgreSQL unavailability skip is promoted to a
failure when those flags are set.

---

## 6. Excluded operations

CI does not:

- train the production model
- download or require IEEE-CIS data
- run `scripts/verify_serving_parity.py`
- require the production serving artifact
- contact a hosted MLflow server or use MLflow secrets
- commit `.env` files, database data, `mlruns/`, `data/`, or `artifacts/`
- add runtime dependencies or change pinned versions
- deploy the application or introduce Kubernetes

Training needs private or local data and writes model binaries. Serving-parity
replay needs the production artifact. MLflow tracking is a local experiment
record, gitignored, and isolated from inference. Those are developer or
training-machine operations. CI proves the committed code, migrations, and
synthetic/integration tests still hold.

---

## 7. Local reproduction

PostgreSQL must be running, and `DATABASE_URL` must point at a server whose
role can create `riskforge_test`. The local default is the same URL CI uses;
see `.env.example`. Start the database with `docker compose up -d` or a local
PostgreSQL 17 install, as documented in `docs/audit.md`.

Use the repository interpreter. The active shell may show both `(.venv)` and
`(base)`; do not trust plain `python`.

```bash
./.venv/bin/python -m ruff --version
./.venv/bin/python -m ruff check .
./.venv/bin/python -m compileall -q .
./.venv/bin/alembic upgrade head
./.venv/bin/alembic check
./.venv/bin/alembic current
./.venv/bin/python -m pytest -q
```

Ruff must report `0.17.3`. `alembic current` must show `7eccd8f63a8c`.
To reproduce the CI rule that database tests cannot skip, set
`RISKFORGE_REQUIRE_POSTGRES=1` in the environment before pytest.
