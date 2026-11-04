# Reproducibility and checks

## Data and artifacts

The IEEE-CIS dataset and fitted model artifacts are not stored in this repository. Obtain the dataset through the Kaggle competition and place the source CSV files under `data/raw/ieee_cis/`.

Generated behavioral data, MLflow runs, and model artifacts are local outputs and are gitignored. The API requires a serving artifact; it returns `503` when the artifact is missing or invalid rather than training a replacement.

## Offline workflow

Run from the repository root:

```bash
PYTHONPATH=. python scripts/inspect_dataset.py
PYTHONPATH=. python scripts/build_behavioral_features.py
PYTHONPATH=. python scripts/evaluate_calibration.py
PYTHONPATH=. python scripts/select_decision_policy.py
PYTHONPATH=. python scripts/build_serving_artifact.py
```

The three-model benchmark and segment/bootstrap reports are generated with:

```bash
PYTHONPATH=. python scripts/run_model_benchmark.py
```

## Local checks

Start PostgreSQL and apply migrations before running the full test suite:

```bash
docker compose up -d
.venv/bin/alembic upgrade head
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall -q .
.venv/bin/alembic check
RISKFORGE_REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -q
```

## Offline/API parity

After building the serving artifact, run:

```bash
PYTHONPATH=. python scripts/verify_serving_parity.py
```

The recorded local verification on 2026-11-04 compared three validation transactions selected by their calibrated probability: one ALLOW, one REVIEW, and one BLOCK. All 28 feature values matched between the offline and API paths. Probabilities, decisions, reason codes, and decision explanations also matched; the largest absolute difference in the two floating-point amount-sum features was 1.310e-10 against a 1e-9 tolerance.

This is a three-case parity check, not an exhaustive proof for every possible request. It does not verify that caller-supplied history is complete or authentic.

## CI scope

GitHub Actions runs Ruff, Python compilation, Alembic migration/schema checks, and pytest against PostgreSQL 17.

CI does not download IEEE-CIS, train a model, build the serving artifact, run the real-data parity script, connect to MLflow, or deploy the API. Those steps require local data and/or generated artifacts.

## Environment

The repository targets Python 3.12 and PostgreSQL 17. Use the committed `.env.example` only for local development; replace its development credentials outside a local environment.
