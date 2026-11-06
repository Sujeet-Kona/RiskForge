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
