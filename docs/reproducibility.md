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
cp .env.example .env
```

The committed requirements file pins the Python packages used for the project. Keep the virtual environment active for the commands below.

## Dataset inputs

The IEEE-CIS Fraud Detection dataset is not committed to this repository. Obtain it through its Kaggle competition page and place both source files at these exact paths:

```text
data/raw/ieee_cis/
├── train_transaction.csv
└── train_identity.csv
```

Do not rename the files: the inspection and feature-generation scripts expect these paths. The generated behavioral Parquet file, fitted serving bundle, and MLflow outputs are local artifacts and are not committed.

Check that both CSV files are present, then inspect the source schema and target distribution:

```bash
PYTHONPATH=. python scripts/inspect_dataset.py
```

This command fails early if either required CSV is missing.

## Offline feature, benchmark, and artifact workflow

Run each stage from the repository root, in this order:

```bash
# 1. Build point-in-time behavioral features from train_transaction.csv
PYTHONPATH=. python scripts/build_behavioral_features.py

# 2. Train and compare the three fixed candidate pipelines on the
#    chronological split; writes model, segment, bootstrap, and metadata reports
PYTHONPATH=. python scripts/run_model_benchmark.py

# 3. Select the probability calibrator using forward-chaining VALIDATION scores
PYTHONPATH=. python scripts/evaluate_calibration.py

# 4. Run validation-only decision-policy research
PYTHONPATH=. python scripts/select_decision_policy.py

# 5. Build the frozen local serving bundle
PYTHONPATH=. python scripts/build_serving_artifact.py
```

The benchmark writes:

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
documented in [temporal evaluation](temporal_evaluation.md). It is separate
from the locked benchmark and requires the same local IEEE-CIS inputs.

```bash
PYTHONPATH=. python scripts/run_temporal_evaluation.py
```

## Local API and database checks

Start PostgreSQL 17 and apply migrations:

```bash
docker compose up -d
.venv/bin/alembic upgrade head
```

Run static, migration, and test checks:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall -q .
.venv/bin/alembic check
RISKFORGE_REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -q
```

Start the API only after building the serving artifact:

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

This is a three-case parity check, not an exhaustive proof for every possible
request. It does not verify that caller-supplied history is complete or
authentic.

## CI scope

GitHub Actions runs Ruff, Python compilation, Alembic migration/schema checks,
and pytest against PostgreSQL 17.

CI does **not** download IEEE-CIS, train a model, build the serving artifact,
run the real-data parity script, connect to MLflow, or deploy the API. Those
steps require the separately obtained dataset and/or generated local artifacts.
A green CI run verifies the committed application and database test suite; it
is not a fresh model-training or benchmark reproduction.

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
| `reports/temporal_robustness/` (5 files) | **yes** | check the walk-forward and bootstrap numbers |
| `reports/calibration/evaluation_metrics.json` | **no** — generated on demand | verify relative Brier/ECE reductions |
| `data/raw/ieee_cis/*.csv` | **no** — gitignored, obtain from Kaggle | run any training or evaluation script |
| `data/processed/train_behavioral_v1.parquet` | **no** — gitignored | skip stage 1 of the workflow |
| `artifacts/serving/riskforge_serving_artifact.joblib` | **no** — gitignored | serve predictions or run the parity check |
| `mlruns/`, `mlflow.db` | **no** — gitignored | reproduce experiment tracking |
| `.env` | **no** — gitignored | supply local database and reviewer-key settings |

Two consequences worth stating plainly:

- **The test suite does not need the dataset or the artifact.** The fixtures
  build a synthetic 600-row modeling frame through the real
  `build_modeling_frame` and the real preprocessor (`tests/conftest.py:46-56`),
  so the full suite — including the serving-discipline, explainability,
  parity, migration, and concurrency suites — run and pass on a clean clone with
  only a PostgreSQL server available. This is why CI is green without IEEE-CIS.
- **Serving does need them.** `app/dependencies.py:80-109` fails closed when
  the artifact is missing or invalid; there is no fallback model. To run the
  API locally you must obtain the dataset and execute stages 1 through 5 of the
  workflow above, which regenerates the behavioral Parquet file and the frozen
  serving bundle.

The committed `reports/` CSVs are historical generated artifacts. They can be
checked for internal consistency and against the code that produced them
without the dataset, but re-deriving their exact floats requires the IEEE-CIS
inputs and the pinned environment.
