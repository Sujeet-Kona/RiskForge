# RiskForge

[![RiskForge CI](https://github.com/Sujeet-Kona/RiskForge/actions/workflows/ci.yml/badge.svg)](https://github.com/Sujeet-Kona/RiskForge/actions/workflows/ci.yml)

**A reproducible fraud-risk scoring research project** built around point-in-time
features, chronological evaluation, probability calibration, cost-sensitive
decisions, model explanations, a FastAPI inference service, and PostgreSQL audit
records.

RiskForge uses the IEEE-CIS Fraud Detection dataset as its offline research
benchmark. It is a portfolio / engineering project, not a deployed payment
network fraud system and not a claim of production readiness.

## What it does

Given a transaction and the strictly-prior behavioral history supplied by the
caller, RiskForge:

1. Builds a fixed 28-feature model input, including rolling card and address
   activity features.
2. Scores it with a fitted XGBoost pipeline.
3. Converts the raw score with an offline-fitted isotonic calibrator.
4. Applies a frozen decision policy and returns `ALLOW`, `REVIEW`, or
   `BLOCK`.
5. Explains model signals separately from the decision rule.
6. Writes each returned decision to PostgreSQL before returning it.

The API is stateless with respect to transaction history: it does not maintain
an authoritative history across requests. The caller is responsible for
supplying complete and trustworthy prior history. The audit table is not used
as a feature store and does not contain the prior-history rows needed to replay
a score.

## Architecture

```text
IEEE-CIS data
    |
    v
Data validation and chronological train / validation / test split
    |
    v
Point-in-time behavioral feature engineering
    |
    v
Train-only preprocessing + XGBoost
    |
    v
Validation-only probability calibration
    |
    v
Frozen cost-sensitive decision policy
    |
    v
SHAP model signals + separate policy reason
    |
    v
FastAPI inference service
    |
    v
PostgreSQL decision audit (application-level append-only)
```

The inference path loads a prebuilt artifact. It does not train, fit a
calibrator, or contact MLflow while serving requests.

## Main engineering decisions

- **Point-in-time correctness:** only transactions with an earlier
  `TransactionDT` may contribute to a transaction's behavioral features.
  Transactions sharing a timestamp cannot see one another.
- **Chronological evaluation:** train, validation, and test partitions are
  ordered in time; equal timestamps are kept together.
- **Fit boundaries:** preprocessing and the classifier are fitted on TRAIN;
  calibration and policy selection use VALIDATION; TEST is reserved for final
  evaluation.
- **Explicit policy:** model probability and business decision are separate.
  The cost assumptions are synthetic project assumptions, not measured
  payment-network economics.
- **Explainability:** SHAP-derived model signals are not conflated with the
  threshold rule that produced the final action.
- **Auditability:** a successful API response is returned only after its
  decision record has been written.

## Measured benchmark results

The following results were reproduced locally on the IEEE-CIS benchmark. The
dataset and fitted artifacts are not committed; CI tests the application and
invariants but does not retrain the model.

| Partition | Rows | Fraud cases |
| --- | ---: | ---: |
| TRAIN | 414,446 | — |
| VALIDATION | 88,013 | 3,010 |
| TEST | 88,081 | 3,065 |

The 28-feature XGBoost model produced the following held-out TEST ranking
metrics. Isotonic calibration was selected using forward-chaining validation
OOF Brier score; the calibrated TEST metrics are reported separately from the
raw model scores.

| TEST metric | Raw XGBoost | Isotonic calibrated |
| --- | ---: | ---: |
| PR-AUC | 0.449254 | 0.438919 |
| ROC-AUC | 0.865318 | 0.864850 |
| Brier score | — | 0.024204 |
| Log loss | — | 0.102851 |
| ECE | — | 0.003553 |

Calibration method selection used 70,334 forward-chaining OOF VALIDATION rows;
17,679 initial warm-up rows were excluded. Isotonic achieved the lower OOF
Brier score (0.024703 vs 0.025368 for sigmoid), while sigmoid had better OOF
log loss and ranking metrics. See [model evaluation](docs/model-evaluation.md)
for the complete comparison and caveats.

The repository includes a reproducible three-model benchmark for Logistic Regression, Random Forest and XGBoost, plus TEST temporal/segment analysis and paired bootstrap intervals. The script is committed, but its private-dataset run must be completed locally before any comparative numeric result is added to this README. See [the benchmark protocol](docs/model-evaluation.md) and [technical defense guide](docs/technical-defense.md).

## Repository layout

| Path | Responsibility |
| --- | --- |
| `ml/features/` | Point-in-time feature engineering |
| `ml/split/` | Chronological data partitioning |
| `ml/models/` | Preprocessing, Logistic Regression baseline, XGBoost, calibration, artifact handling |
| `policy/` | Decision policy, cost assumptions, policy evaluation |
| `explain/` | Model attribution and reason-code taxonomy |
| `app/` | FastAPI schemas, inference service, API lifecycle |
| `db/` and `alembic/` | PostgreSQL audit model and migrations |
| `tracking/` | Local MLflow experiment lineage |
| `scripts/` | Data, training, evaluation, artifact, parity and final benchmark workflows |
| `tests/` | Unit, API, database and integration tests |
| `docs/` | Detailed API, audit, CI, explainability and MLflow notes |

## Requirements

- Python 3.12
- PostgreSQL 17 for the full test suite and API audit persistence
- Docker Compose (optional; used to run local PostgreSQL)
- IEEE-CIS Fraud Detection data for the offline data and model workflows

The dataset and generated artifacts are intentionally not committed. Obtain the
IEEE-CIS data through its official Kaggle competition page and place the source
CSV files under `data/raw/ieee_cis/` as expected by the scripts. Generated
behavioral data, MLflow runs and model artifacts remain local and are gitignored.

## Local setup

Clone the repository, then from its root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Start the local PostgreSQL service:

```bash
docker compose up -d
```

The committed `.env.example` contains development-only database credentials.
Do not use those defaults outside a local environment. Configure
`DATABASE_URL` in your local `.env` if your PostgreSQL settings differ.

Apply the audit schema:

```bash
.venv/bin/alembic upgrade head
```

## Data and model workflow

After placing the required IEEE-CIS CSV files in `data/raw/ieee_cis/`, run the
offline stages from the repository root:

```bash
PYTHONPATH=. python scripts/inspect_dataset.py
PYTHONPATH=. python scripts/build_behavioral_features.py
PYTHONPATH=. python scripts/evaluate_calibration.py
PYTHONPATH=. python scripts/select_decision_policy.py
PYTHONPATH=. python scripts/build_serving_artifact.py
```

These scripts are explicit research workflows, not an automated production
training service. The serving-artifact builder enforces the locked model and
calibration configuration described in the source and documentation. Keep the
generated files local; do not commit dataset files or fitted model binaries.

For the Logistic Regression baseline diagnostic, run:

```bash
PYTHONPATH=. python scripts/train_logistic_baseline.py
```

For experiment lineage, see [docs/mlflow.md](docs/mlflow.md). For the exact
inference contract and artifact lifecycle, see [docs/api.md](docs/api.md).
Use VALIDATION for model comparisons; do not tune or select models using TEST.
See [model evaluation](docs/model-evaluation.md) for the recorded benchmark.

## Run the inference API

With PostgreSQL running, the migration applied, and the serving artifact built:

```bash
uvicorn app.main:app --reload
```

Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Example request:

```bash
curl -X POST http://127.0.0.1:8000/v1/risk/score \
  -H 'Content-Type: application/json' \
  -d '{
    "transaction": {
      "TransactionID": "demo-001",
      "TransactionDT": 1000000,
      "TransactionAmt": 42.50,
      "card1": 12345
    },
    "history": [],
    "top_k": 3
  }'
```

The example is structurally valid; a meaningful score requires a locally built
serving artifact. The API returns `503` if the artifact is missing or invalid.
Every successful decision is persisted before the response is returned.

## Tests and CI

Run the checks from the repository root with PostgreSQL available:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall -q .
.venv/bin/alembic upgrade head
.venv/bin/alembic check
RISKFORGE_REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -q
```

GitHub Actions runs Ruff, Python compilation, Alembic migration/schema checks,
and the full pytest suite against PostgreSQL 17. CI validates the committed
application and tests; it does not download IEEE-CIS data, train the model,
rebuild the serving artifact, or deploy the API. See [docs/ci.md](docs/ci.md).

## Evaluation and limitations

RiskForge separates measured dataset results from project assumptions:

- IEEE-CIS metrics describe performance on the held-out chronological
  benchmark partition, not future production traffic.
- The false-positive, false-negative and review costs used for policy selection
  are synthetic assumptions. The resulting thresholds are experimental.
- The active serving policy remains `policy-001`
  (`T_REVIEW = 0.075`, `T_BLOCK = 0.110`). The corrected forward-chaining
  policy-selection workflow produces an experimental `candidate`; it does not
  change the serving policy. Candidate thresholds have not been promoted or
  revalidated as the active policy. See
  [calibration methodology](docs/calibration.md).
- The service relies on caller-provided history and cannot independently
  guarantee that history is complete, current or authentic.
- Audit records preserve the scored transaction inputs and decision details,
  but not the full prior-history rows; they are not sufficient to replay a score.
- Audit append-only behaviour is enforced by application convention, not database
  privileges or triggers. Request idempotency and duplicate suppression are not
  implemented; repeated transaction IDs can produce multiple audit rows.
- The repository does not provide authentication, TLS termination, rate
  limiting, production secrets management, deployment automation, drift
  monitoring, or an operational review workflow.

These constraints are intentional and documented. RiskForge is an engineering
demonstration of a fraud-risk ML lifecycle and serving boundary, not a certified
fraud-control product.

## Detailed documentation

- [Inference API](docs/api.md)
- [Model evaluation, three-model benchmark and robustness protocol](docs/model-evaluation.md)
- [Technical defense guide](docs/technical-defense.md)
- [Calibration methodology](docs/calibration.md)
- [Audit persistence](docs/audit.md)
- [Explainability](docs/explainability.md)
- [MLflow lifecycle tracking](docs/mlflow.md)
- [CI design and verification scope](docs/ci.md)

## Resume-ready project summary

**RiskForge — Fraud Risk Scoring & Decision Engine**

- Built a reproducible fraud-risk ML pipeline over 590,540 IEEE-CIS transactions using 28 point-in-time behavioral features, chronological evaluation, XGBoost, forward-chaining probability calibration, and SHAP-based explanations.
- Evaluated the calibrated model on a held-out chronological TEST partition, achieving **0.4389 PR-AUC**, **0.8649 ROC-AUC**, **0.0242 Brier score**, and **0.0036 ECE**. Calibration method selection used forward-chaining VALIDATION OOF predictions; see the evaluation report for trade-offs and limitations.
- Implemented a FastAPI scoring service with a versioned model artifact and PostgreSQL decision auditing; manually verified offline/API parity on three representative ALLOW, REVIEW, and BLOCK cases, with all 28 features, probabilities, decisions, and reason codes matching.

These bullets describe the current repository and measured benchmark only. They do not imply a deployed payment-network system, production traffic performance, an authoritative transaction-history service, or database-enforced immutable audit storage.

## Release verification

The manual release checks and the latest observed offline/API parity result are recorded in [release verification](docs/release-verification.md). CI status is available from the badge at the top of this README; CI does not train on IEEE-CIS or run the artifact parity replay.
