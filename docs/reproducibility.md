# Reproducibility and checks

## Environment

RiskForge targets **Python 3.12** and **PostgreSQL 17**. Start from a clean checkout:

```bash
git clone https://github.com/Sujeet-Kona/RiskForge.git
cd RiskForge

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ruff==0.17.3
[ -f .env ] || cp .env.example .env
```

The committed requirements file pins the Python packages used for the project.
Keep the virtual environment active for the commands below. Review the root
`.env` before sourcing it; Python and Alembic do **not** automatically load it.
The database section below exports it before any database commands.

## Dataset inputs

The IEEE-CIS Fraud Detection dataset is not committed to this repository. Obtain it through its Kaggle competition page and place both source files at these exact paths:

```text
data/raw/ieee_cis/
├── train_transaction.csv
└── train_identity.csv
```

Do not rename the files: the inspection and feature-generation scripts expect these paths. The generated behavioral Parquet file, fitted serving bundle, and MLflow outputs are local artifacts and are not committed.

Optionally inspect the source schema and target distribution:

```bash
PYTHONPATH=. python scripts/inspect_dataset.py
```

Inspection requires both CSVs and fails if either is missing. The minimal
serving build below reads `train_transaction.csv`; it does not use the identity
CSV or require the inspection script.

## Minimal serving build

From the repository root, with the source data available:

```bash
PYTHONPATH=. python scripts/build_behavioral_features.py
PYTHONPATH=. python scripts/build_serving_artifact.py
```

These commands generate the behavioral Parquet file and fit the locked
preprocessing/XGBoost pipeline on TRAIN plus the locked isotonic calibrator on
VALIDATION, then write the local serving bundle. This is an offline model fit,
not a download of a prebuilt model; rerunning replaces generated features and
the bundle. No benchmark, calibration-evaluation, policy-search, or temporal
report is a prerequisite. The API only loads the resulting bundle.

## Optional research and report regeneration

These are separate experiments, **not serving setup steps**. Run only when you
intend to refit models and regenerate research outputs; scripts may overwrite
existing reports. Generated calibration outputs are not supplied by a clean
checkout and are not produced by the minimal serving build.

```bash
# Compare candidate models and regenerate benchmark reports
PYTHONPATH=. python scripts/run_model_benchmark.py

# Evaluate calibration methods and generate calibration outputs
PYTHONPATH=. python scripts/evaluate_calibration.py

# Run validation-only decision-policy research
PYTHONPATH=. python scripts/select_decision_policy.py
```

When run successfully, the benchmark writes or replaces:

- `reports/model_benchmark/model_metrics.csv`
- `reports/model_benchmark/test_segment_metrics.csv`
- `reports/model_benchmark/test_bootstrap_intervals.csv`
- `reports/model_benchmark/run_metadata.json`

New benchmark runs record SHA-256 fingerprints and byte sizes for the source
inputs, Python/platform details, key numerical-library versions, and the Git
commit when available. The committed report files are historical outputs from
a local run; the committed `run_metadata.json` predates the provenance fields
and does not yet contain source-file fingerprints. A fresh local benchmark run
with the private dataset files will generate updated metadata. The dataset
itself must not be committed.

## Additional temporal robustness checks

The additional expanding-window and time-block bootstrap experiment is
documented in [temporal evaluation](temporal_evaluation.md). It is optional,
refits models, and can overwrite six files under `reports/temporal_robustness/`.
It requires the local IEEE-CIS inputs and generated behavioral features.

```bash
PYTHONPATH=. python scripts/run_temporal_evaluation.py
```

## Local API and database checks

Use a **disposable LOCAL PostgreSQL 17 server only**, not a remote, shared, or
existing user database. Review the root `.env`: `DATABASE_URL` must point to
that server on `localhost` or `127.0.0.1`, with port, database, and role matching
`POSTGRES_PORT`, `POSTGRES_DB`, and `POSTGRES_USER`. If the default port is in
use, choose a free local port and update both settings. A loopback address alone
does not prove isolation; ensure it is this container, not a tunnel.

Compose reuses its named volume and fixed `riskforge-postgres` container name;
starting it does not guarantee an empty database. If existing state is not
explicitly disposable, stop and provision an isolated local server instead.
Do not run migrations against an existing/user database just to make setup pass.

Export the reviewed environment **before Alembic or Python**, then wait for the
container healthcheck (requires Docker Compose with `--wait` support):

```bash
set -a
source .env
set +a
docker compose up -d --wait postgres
docker compose port postgres 5432
```

Stop on any error. Confirm the published port matches `DATABASE_URL`, then run
this read-only target check from the repository root:

```bash
python - <<'PY'
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from db.config import DatabaseConfig

if not os.environ.get("DATABASE_URL"):
    raise SystemExit("Stop: export the reviewed DATABASE_URL first.")
url = make_url(DatabaseConfig.from_environment().url)
expected = (os.environ["POSTGRES_USER"], os.environ["POSTGRES_DB"], int(os.environ["POSTGRES_PORT"]))
if (url.host not in {"localhost", "127.0.0.1"} or url.query
        or (url.username, url.database, url.port or 5432) != expected):
    raise SystemExit("Stop: DATABASE_URL does not match the local Compose target.")
print(f"Target: {url.host}:{url.port or 5432}/{url.database}; role: {url.username}")
with create_engine(url).connect() as connection:
    print(connection.execute(text("SELECT current_database(), current_user")).one())
PY
```

Only after the output confirms the intended disposable database and role,
apply migrations in this same shell:

```bash
.venv/bin/alembic upgrade head
.venv/bin/alembic check
```

Static checks do not require a database:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall -q .
```

**Before pytest:** `tests/conftest.py` always derives `riskforge_test` on the
server configured by `DATABASE_URL` (or the default server if unset), regardless
of the database name you supply. It may create that database, downgrade it to
base, rebuild its schema, and truncate tables. Changing only the URL's database
name does not isolate tests. Use only the verified disposable LOCAL server,
and ensure any existing `riskforge_test` data can be destroyed. Set
`RISKFORGE_REQUIRE_POSTGRES=1` so unavailable PostgreSQL fails rather than
silently skipping database coverage:

```bash
RISKFORGE_REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -q
```

Start the API in the same exported environment only after the minimal serving
build and verified database setup:

```bash
uvicorn app.main:app --reload
```

The interactive API schema is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The API requires the local fitted artifact. If it is missing, invalid, or
incompatible, scoring is unavailable; the service does not train a replacement
inside a request.

## Offline/API parity

After building the serving artifact and preparing the dataset-derived
behavioral file, run:

```bash
PYTHONPATH=. python scripts/verify_serving_parity.py
```

The recorded local verification on 2026-11-04 compared three VALIDATION
transactions selected by calibrated probability: one ALLOW, one REVIEW, and
one BLOCK. All 28 feature values matched between the offline and API paths.
Probabilities, decisions, reason codes, and decision explanations also
matched; the largest absolute difference in the two floating-point amount-sum
features was 1.310e-10 against a 1e-9 tolerance.

The check was rerun successfully during local hardening on 2026-11-15; see
[verification evidence](verification.md). Its audit-store double keeps records
in memory and never writes parity decisions to PostgreSQL.

This is a three-case parity check, not an exhaustive proof for every possible
request. It does not verify that caller-supplied history is complete or
authentic.

## CI scope

GitHub Actions runs Ruff, Python compilation, Alembic migration/schema checks,
and pytest against PostgreSQL 17. The frontend job runs `npm ci` from the
committed lockfile, the Vitest suite (`npm test`), and `npm run build`, which
type-checks before the production build. See [frontend checks](../frontend/README.md#frontend-checks)
for local commands.

A separate Playwright job uses Chromium, real FastAPI, Alembic and a disposable
`riskforge_e2e` PostgreSQL database for reviewer authentication and resolution.
There is no API mocking in that suite. See the [E2E command and database guards](../frontend/README.md#browser-review-e2e).

CI does **not** download IEEE-CIS, train a model, build the serving artifact,
run real-data parity, connect to MLflow, or deploy the API. Those steps require
separately obtained data and generated artifacts. The browser suite seeds
explicitly synthetic decisions; it does not establish real model-scoring parity.
A green CI run is not a fresh benchmark reproduction, independent human review,
or production-readiness approval.

## Reproducibility boundaries

- The benchmark reports one chronological split of the public IEEE-CIS data.
- The source dataset is external and is not redistributed by this repository.
- The saved benchmark CSVs are committed evidence; rerunning them requires the
  exact source files and the pinned environment.
- Runtime and source fingerprints are emitted by new benchmark runs; they are
  not retroactively available in the currently committed metadata.
- The offline/API parity evidence covers three selected validation cases only.

## Seeding, threads, and what "reproducible" means here

Every source of randomness in the pipeline takes an explicit seed at the point
of use. There is no process-wide seeding anywhere in the repository: no
`np.random.seed`, no `random.seed`, and no `PYTHONHASHSEED` setting.

| Source of randomness | Fixed value | Where | What it controls |
| --- | --- | --- | --- |
| XGBoost sampling | `random_state=42` | `ml/models/xgboost_model.py:85` | `subsample=0.8` and `colsample_bytree=0.8` on the locked model |
| Benchmark bootstrap | `np.random.default_rng(20261104)` | `scripts/run_model_benchmark.py:47,266` | 500 paired resamples; the seed is also written into `run_metadata.json` |
| Temporal bootstrap | `np.random.default_rng(20261107)` | `scripts/run_temporal_evaluation.py:38,162` | 500 moving-time-block resamples; also written into `run_metadata.json` |
| RandomForest comparison | `random_state=42` | `scripts/run_model_benchmark.py:128` | benchmark-only candidate, never served |
| LogisticRegression baseline | `random_state=42` | `ml/models/logistic_baseline.py:65` | baseline candidate, never served |
| `sigmoid` calibrator | `random_state=42` | `ml/models/calibration.py:101` | the **non**-locked calibration method |

Three things in the locked path consume no randomness at all, which is worth
stating because it narrows what a rerun can legitimately differ on:

- The chronological split (`ml/split/temporal.py`) selects boundaries from
  distinct `TransactionDT` values. It contains no RNG call.
- The locked calibrator is `IsotonicRegression` (`ml/models/calibration.py:104`),
  fitted by the deterministic PAVA algorithm.
- The behavioral features are built by a single ordered pass over
  `TransactionDT`, with no sampling.

All Python packages are `==`-pinned in `requirements.txt`, and the pinned
versions match the environment the committed reports were produced in
(`pandas 3.1.0`, `numpy 2.5.4`, `scikit-learn 1.9.1`, `scipy 1.18.1`,
`xgboost 3.4.1`).

### The one execution setting that is deliberately not pinned

`n_jobs=-1` at `ml/models/xgboost_model.py:86` and
`scripts/run_model_benchmark.py:129` resolves to the local core count, so the
thread count of a rerun is a property of the machine rather than of the
repository.

This is not an oversight that can be quietly corrected. Fixing `n_jobs` to a
specific number would change the thread count used to build the histogram
gradients, and XGBoost does not guarantee bit-identical trees across thread
counts. The locked benchmark, the locked calibrator, the frozen
`scale_pos_weight`, and the frozen serving artifact were all produced with
`n_jobs=-1` on the original machine. Pinning threads now would risk producing a
*model* that no longer matches the frozen metrics — which is a far worse outcome
than an unpinned thread count, because the frozen numbers are the project's
evidence.

What the repository does instead is **fail loudly on drift** rather than
guarantee identical floats:

| Guard | Tolerance | Where | Refuses to continue when |
| --- | --- | --- | --- |
| Locked VALIDATION/TEST PR-AUC and ROC-AUC | `1e-6` | `scripts/train_tracked_model.py:42-46,107` | a tracked rerun is not the locked model |
| Locked raw and calibrated TEST PR-AUC and ROC-AUC | `1e-5` | `scripts/select_decision_policy.py:51-54,448` | the Phase 3 baseline moved |
| Locked `scale_pos_weight` | `1e-6` | `scripts/build_serving_artifact.py:62,110` | the class weighting moved |

So the honest claim is: **reproduction is asserted, not promised.** On the
pinned stack and the same source data the pipeline is expected to reproduce the
locked figures, and if it does not, the scripts above abort with the observed
and expected values printed instead of writing a report that silently disagrees
with the committed one. A rerun on a machine with a different core count, a
different BLAS, or a different OS is not covered by any evidence in this
repository.

## What a clean clone contains

`.gitignore` excludes the dataset, generated features, the fitted serving
bundle, and MLflow state. This is deliberate — none of them should be
redistributed — but it means a fresh checkout is **not** immediately runnable
as a serving system.

| Path | In a clean clone | Needed to |
| --- | --- | --- |
| `app/`, `db/`, `ml/`, `policy/`, `explain/`, `tracking/`, `scripts/`, `alembic/`, `tests/`, `frontend/src/` | **yes** | read and test the code |
| `docs/`, `README.md`, `.env.example`, `.github/workflows/ci.yml` | **yes** | understand and CI-verify the project |
| `reports/model_benchmark/` (4 files) | **yes** | check the locked benchmark numbers |
| `reports/temporal_robustness/` (6 files) | **yes** | check the walk-forward and bootstrap numbers |
| `reports/calibration/evaluation_metrics.json` | **no** — optional calibration-evaluation output | verify relative Brier/ECE reductions after regeneration |
| `data/raw/ieee_cis/*.csv` | **no** — gitignored, obtain from Kaggle | run real-data training or evaluation |
| `data/processed/train_behavioral_v1.parquet` | **no** — gitignored | build the serving bundle and run research scripts |
| `artifacts/serving/riskforge_serving_artifact.joblib` | **no** — gitignored | serve predictions or run the parity check |
| `mlruns/`, `mlflow.db` | **no** — gitignored | reproduce experiment tracking |
| `.env` | **no** — gitignored | supply local database and reviewer-key settings |

Two consequences worth stating plainly:

- **The test suite does not need the real dataset or serving artifact.** The
  fixtures fit small synthetic models using a 600-row modeling frame and the
  real feature contract/preprocessor. Database tests still require the
  disposable LOCAL server and destructive-test precautions above; no pass
  result is implied merely by installing dependencies.
- **Serving does need a compatible fitted bundle.** The API fails closed when
  it is missing or invalid; there is no fallback model. Obtain the source data
  and run the two commands in [Minimal serving build](#minimal-serving-build)
  to generate behavioral features and the local bundle. Optional research
  reports and calibration-evaluation outputs are not required.

The committed `reports/` CSVs are historical generated artifacts. They can be
checked for internal consistency and against the code that produced them
without the dataset, but re-deriving their exact floats requires the IEEE-CIS
inputs and the pinned environment.
