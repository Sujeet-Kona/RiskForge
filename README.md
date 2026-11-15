# RiskForge

[![CI](https://github.com/Sujeet-Kona/RiskForge/actions/workflows/ci.yml/badge.svg)](https://github.com/Sujeet-Kona/RiskForge/actions/workflows/ci.yml)

**An end-to-end fraud-risk modeling and transaction decisioning system** built with Python, XGBoost, FastAPI, and PostgreSQL.

RiskForge uses the public IEEE-CIS Fraud Detection dataset to build point-in-time behavioral features, evaluate models on chronological partitions, calibrate predicted probabilities, apply an explicit decision policy, and serve transaction scores with TreeSHAP signals and audit records.

> **Scope:** This is an offline research and portfolio system evaluated on historical public data. It is not integrated with a payment network, and the reported metrics do not establish live fraud performance.

## At a glance

| Item | Implementation |
|---|---|
| Dataset | IEEE-CIS Fraud Detection |
| Transactions | 590,540 |
| Fraud cases | 20,663 (3.50%) |
| Model feature contract | 28 features: 13 behavioral, 4 categorical, 11 C/D |
| Temporal partitions | 414,446 train / 88,013 validation / 88,081 test |
| Selected estimator | XGBoost |
| Calibration | Isotonic regression selected using forward-chaining validation; method persisted with each decision |
| Test Average Precision | 0.449254 raw / 0.438919 calibrated |
| Test ROC-AUC | 0.865318 raw / 0.864850 calibrated |
| Decision outputs | ALLOW / REVIEW / BLOCK |
| Serving and audit | FastAPI / PostgreSQL |

Average Precision is reported using scikit-learn's average_precision_score. It is the primary ranking metric for this imbalanced dataset.

## System architecture

The offline path builds and evaluates a frozen artifact. The online path loads that artifact and scores requests; it does not train or tune a model.

~~~mermaid
flowchart TB
  subgraph OFFLINE[Offline model development]
    DATA[IEEE-CIS data] --> FEATURES[Point-in-time feature builder]
    FEATURES --> SPLIT[Chronological 70 / 15 / 15 split]
    SPLIT --> TRAIN[Train preprocessing + XGBoost]
    SPLIT --> VAL[Forward-chaining validation]
    TRAIN --> MODEL[Frozen model pipeline]
    VAL --> CAL[Select and fit calibrator]
    MODEL --> ARTIFACT[Versioned serving artifact]
    CAL --> ARTIFACT
  end
  subgraph ONLINE[FastAPI inference]
    REQUEST[Transaction + prior history] --> VALIDATE[Strict request validation]
    VALIDATE --> BUILD[Shared 28-feature builder]
    BUILD --> SCORE[XGBoost score]
    SCORE --> CALIBRATE[Isotonic calibration]
    CALIBRATE --> POLICY[policy-001]
    SCORE --> SHAP[TreeSHAP signals]
    POLICY --> AUDIT[(PostgreSQL decision record)]
    SHAP --> AUDIT
    AUDIT --> RESPONSE[Scoring response]
    AUDIT -->|REVIEW only| CASE[(Review case)]
    CASE --> REVIEWER[Authenticated reviewer]
    REVIEWER --> EVENTS[(Append-only review event)]
    AUDIT --> IDEMPOTENCY[(Idempotency response record)]
  end
  ARTIFACT -. loaded at startup .-> SCORE
  ARTIFACT -. calibrator .-> CALIBRATE
~~~

### Design boundaries

- Offline feature generation and online scoring use the same behavioral feature implementation.
- The API loads a frozen local artifact at startup; it does not train a fallback model during requests.
- Each scoring response, decision-history row, and review case records the artifact's calibration method alongside model, feature-contract, and policy identifiers.
- Prior transaction history is supplied by the caller. PostgreSQL stores decisions but is not used as a feature-history store.
- TreeSHAP model signals and policy-threshold explanations are separate outputs.
- MLflow is used for offline experiment tracking only; it is not called in the inference path.

## Dataset and point-in-time features

The modeling frame contains 590,540 transactions and a fixed 28-feature input contract.

| Feature family | Count | Examples |
|---|---:|---|
| Behavioral | 13 | Amount transform, elapsed-time buckets, card/address 1-hour and 24-hour counts, history flags, 24-hour amount sums |
| Categorical | 4 | ProductCD, card6, P_emaildomain, R_emaildomain |
| C/D numeric | 11 | Selected C* and D* fields |
| **Total** | **28** | Locked model input contract |

The entity keys card1, addr1, and addr2 construct history but are not model features. TransactionDT is used for temporal ordering and splitting, not as a model feature.

### Leakage controls

- A transaction can use only events with TransactionDT strictly earlier than its own timestamp.
- Rows sharing a timestamp are processed as one group and cannot see one another.
- The trailing 1-hour and 24-hour windows use an exclusive lower boundary.
- Input rows must be sorted by TransactionDT.
- Chronological partitions are made over distinct timestamps; tied timestamps are not split across partitions.
- Preprocessing and estimators are fitted on TRAIN. Calibration and policy research use VALIDATION. TEST is reserved for final reporting.

### Chronological partitions

| Partition | Transactions | Fraud cases | Fraud rate |
|---|---:|---:|---:|
| TRAIN | 414,446 | 14,588 | 3.52% |
| VALIDATION | 88,013 | 3,010 | 3.42% |
| TEST | 88,081 | 3,065 | 3.48% |
| **Total** | **590,540** | **20,663** | **3.50%** |

## Model evaluation

### Candidate comparison on TEST

Each candidate's preprocessing and estimator were fitted on TRAIN only.

| Model | Average Precision | ROC-AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.145789 | 0.780255 | 0.178173 | 0.538330 |
| Random Forest | 0.426033 | 0.861220 | **0.077925** | **0.296063** |
| XGBoost | **0.449254** | **0.865318** | 0.101228 | 0.344462 |

XGBoost had the highest Average Precision and ROC-AUC. Random Forest had the better raw Brier score and log loss; XGBoost was not best on every metric.

Saved reports: [model metrics](reports/model_benchmark/model_metrics.csv) · [test segment metrics](reports/model_benchmark/test_segment_metrics.csv) · [bootstrap intervals](reports/model_benchmark/test_bootstrap_intervals.csv).

### Paired bootstrap comparison

A paired, class-stratified percentile bootstrap with 500 replicates compared XGBoost with Random Forest on the same TEST transactions.

| Difference (XGBoost − Random Forest) | Estimate | 95% bootstrap interval |
|---|---:|---:|
| Average Precision | +0.023220 | +0.015090 to +0.030551 |
| ROC-AUC | +0.004098 | +0.001328 to +0.007241 |

The bootstrap resamples transactions, not time blocks. It does not account for temporal dependence, dataset shift, retraining variability, or threshold-selection uncertainty.

### Probability calibration

The calibrator was selected on forward-chaining validation predictions: five chronological folds by distinct timestamps, with the first fold used as warm-up. Later folds were scored using calibrators fitted only on earlier folds. The comparison used 70,334 out-of-fold validation rows.

| Calibration method | Validation OOF Brier | OOF log loss |
|---|---:|---|
| Sigmoid | 0.025368 | Lower than isotonic |
| Isotonic | **0.024703** | Higher than sigmoid |

Isotonic was selected using Brier score as the selection criterion; it did not win every calibration metric. The selected calibrator was refitted on all VALIDATION scores and evaluated once on TEST.

| Metric | Raw XGBoost | Isotonic-calibrated XGBoost |
|---|---:|---:|
| Average Precision | 0.449254 | 0.438919 |
| ROC-AUC | 0.865318 | 0.864850 |
| Brier score | 0.101228 | **0.024204** |
| Log loss | 0.344462 | **0.102851** |
| Expected Calibration Error (10 equal-width bins) | Not included in saved benchmark CSV | 0.003553 |

The TEST Brier score decreased by approximately **76.1%**. Calibration improves probability quality, not ranking; the small reductions in Average Precision and ROC-AUC are consistent with isotonic calibration creating tied scores. The raw TEST ECE baseline is not included in the committed benchmark CSV, so no percentage ECE reduction is claimed here.

### Segment analysis and temporal evaluation

The benchmark reports descriptive TEST slices by chronological period, transaction-amount bands, ProductCD, and card-history groups. These are slices of the same holdout, not independent validation datasets. See the [segment report](reports/model_benchmark/test_segment_metrics.csv).

An additional expanding-window evaluation, paired moving-time-block bootstrap, and segmented calibration reliability reports are documented in [temporal evaluation](docs/temporal_evaluation.md) and committed under [reports/temporal_robustness/](reports/temporal_robustness/). Across four expanding folds XGBoost averaged **0.4940** Average Precision (std 0.0524, range 0.4244–0.5449) and **0.8884** ROC-AUC, against Random Forest at 0.4629 and 0.8834.

These walk-forward figures are **not comparable to the locked TEST benchmark above** and must not be quoted alongside it. The expanding window overtakes the locked split boundaries: fold 3 trains on 58,741 rows of the locked VALIDATION period, and fold 4's training window includes 29,419 rows — 33.4% — of the locked 88,081-row TEST partition. They are repeated historical simulations on one finite dataset, not independent out-of-sample estimates. Regenerating them requires the gitignored IEEE-CIS source data; the committed CSVs are historical artifacts whose configuration and internal arithmetic have been checked against the generating code, but whose exact floats have not been re-derived here.

## Decision policy

The serving path applies the frozen policy-001 thresholds to calibrated probability:

| Action | Calibrated probability |
|---|---|
| ALLOW | p < 0.075 |
| REVIEW | 0.075 ≤ p < 0.110 |
| BLOCK | p ≥ 0.110 |

Policy research used synthetic assumptions: missed-fraud cost 100 units, false-decline cost 10 units, manual-review cost 5 units, and 70% assumed fraud capture by review. These are experimental inputs, not measured payment-network economics. The policy is amount-blind and should not be interpreted as a validated real-world payment policy.

The original rationale for the 0.075 REVIEW threshold is **not recorded in this repository**, and the cost-sensitive search described above selects 0.050, not 0.075. That gap is documented explicitly rather than retrofitted with a justification — see [policy provenance](docs/model-evaluation.md#provenance-of-the-frozen-thresholds) and the [machine-readable provenance record](reports/policy_provenance.json). `policy-001` is treated as immutable absent a separate, evidence-backed experiment.

## Inference API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Service, artifact identity, and database readiness |
| POST | /v1/risk/score | Score one transaction using caller-supplied prior history; optional `Idempotency-Key` supports safe retries |
| GET | /v1/audit/decisions | Paginated, newest-first decision history; optional ALLOW/REVIEW/BLOCK filter |
| GET | /v1/reviews | Authenticated pending review queue |
| POST | /v1/reviews/{case_id}/resolve | Record an APPROVED or REJECTED human outcome without rewriting the model decision |

The request includes the current transaction and a history array. History timestamps must be strictly earlier than the current transaction; same-timestamp and future records are rejected. Only the trailing 24 hours affect behavioral features. The API accepts at most 2,000 history records and rejects larger requests rather than truncating them. top_k controls returned reason signals (default 3).

The service loads a local artifact identified as xgb-001 / behavioral-v1. It contains preprocessing, XGBoost, the isotonic calibrator, and feature/model/policy metadata. A scoring request validates input, builds the 28-feature row, scores and calibrates it, applies policy-001, creates TreeSHAP and policy explanations, and persists the decision to PostgreSQL before returning success. REVIEW decisions also create a separate pending review case in the same database transaction. Missing or incompatible artifacts fail closed; audit-write failure prevents a successful response.

### Idempotency

The scoring endpoint accepts an optional `Idempotency-Key` header (8–128
characters from letters, digits, `.`, `_`, `:`, and `-`). RiskForge hashes
the canonical validated request body. Repeating the same key with the same
request returns the original stored response without creating a second decision.
Reusing that key for a different request returns HTTP 409. The decision, any
REVIEW case, and the response snapshot are committed in one PostgreSQL
transaction; a concurrent duplicate is resolved by the unique key constraint.

Keys are global to this single-operator prototype and retained indefinitely.
A production multi-tenant service would scope keys to an authenticated tenant
and define an explicit retention/expiry policy.

### Human review

A reviewer queue is available for automated REVIEW decisions. Configure
`RISKFORGE_REVIEWER_API_KEY` and `RISKFORGE_REVIEWER_ID` in the API environment.
The API requires the key in the `X-Reviewer-API-Key` header for queue reads and
resolution actions. This is a deliberately small, single-reviewer local demo
credential, not a multi-user identity provider.

A reviewer may mark a case APPROVED or REJECTED once, with an optional comment.
The automated decision remains REVIEW in the original decision record. The
mutable review-case state and append-only review event are stored separately.
PostgreSQL triggers reject UPDATE/DELETE against the original risk decision and
review-event tables; they do not make the database tamper-proof against a
database administrator.

To configure the local credential, fill the empty `RISKFORGE_REVIEWER_API_KEY=`
line in the git-ignored root `.env` with a random key without printing it, and
restrict the file to your user:

```bash
python -c "import pathlib,re,secrets; p=pathlib.Path('.env'); p.write_text(re.sub(r'(?m)^RISKFORGE_REVIEWER_API_KEY=.*$', 'RISKFORGE_REVIEWER_API_KEY=' + secrets.token_urlsafe(32), p.read_text()))"
chmod 600 .env
```

The API does not load `.env` itself. Export it in the shell that starts the API,
and restart the API after changing either reviewer variable:

```bash
set -a
source .env
set +a
uvicorn app.main:app --reload
```

To enter the key in the dashboard, read it locally from `.env`
(`grep '^RISKFORGE_REVIEWER_API_KEY=' .env | cut -d= -f2-`). Never commit it or
paste it into issues, logs or chat. Without both variables the review endpoints
return 503; a missing or wrong key returns 401.


TreeSHAP contributions are calculated on the raw XGBoost output margin (log-odds) and mapped back to the original 28 contract features. They are local model signals, not causal explanations, and are not calibrated-probability contributions. The policy explanation separately identifies the probability threshold band that produced the action.

## Local dashboard

A React + TypeScript dashboard is available under `frontend/`. It reads
`GET /health` and submits transactions to the existing
`POST /v1/risk/score` endpoint. It displays the calibrated probability, the
policy decision, TreeSHAP reason signals, risk-reducing signals, and artifact
version identifiers. The browser does not reimplement feature construction,
calibration, explainability, or policy logic.

After starting PostgreSQL, building the serving artifact, and starting FastAPI
as described in [Run locally](#run-locally), start the dashboard in another
terminal (Node.js 22.13+ on 22.x, or a compatible supported 24+ runtime):

```bash
cd frontend
[ -f .env ] || cp .env.example .env
npm ci
npm run dev
```

Open http://127.0.0.1:5173. The Vite development proxy forwards `/api` requests
to FastAPI at http://127.0.0.1:8000 by default. See
[frontend setup](frontend/README.md) for API configuration, secret handling,
local typecheck/test/build commands, and deployment boundaries.

The dashboard supports live scoring, result inspection, persisted decision
history, and an authenticated manual-review queue. The UI does not display
fabricated history or operational metrics.

## Verification

The recorded local offline-to-API parity check compared three real VALIDATION transactions: one ALLOW, one REVIEW, and one BLOCK. All 28 feature values matched between offline and API paths. Probabilities, decisions, reason codes, and policy explanations also matched. The largest absolute difference in the two floating-point amount-sum features was 1.310 × 10⁻¹⁰, within the 10⁻⁹ absolute tolerance.

This is a targeted three-case parity check, not exhaustive proof for all possible inputs or caller-supplied histories. See the [verification record](docs/verification.md) for evidence and limitations.

**Before running pytest:** `tests/conftest.py` always derives `riskforge_test` on the configured database server, regardless of the database name in `DATABASE_URL`. It may create, downgrade, rebuild, and truncate that database. Use a disposable **LOCAL** server only, verify the target first, and set `RISKFORGE_REQUIRE_POSTGRES=1` to prevent silent database-test skips. Follow the [database safety and check instructions](docs/reproducibility.md#local-api-and-database-checks).

CI runs Ruff, Python compilation, Alembic migration/schema checks, and pytest with PostgreSQL 17. The frontend job installs dependencies from the committed lockfile with `npm ci`, runs the Vitest suite with `npm test`, then type-checks and creates the production build with `npm run build`. The frontend tests mock API calls; CI does not run a browser end-to-end suite. CI does not download IEEE-CIS, train models, build the serving artifact, run the real-data parity script, connect to an MLflow server, or deploy the API.

## Technology

Python 3.12 · pandas · NumPy · PyArrow · scikit-learn · XGBoost · SHAP/TreeSHAP · FastAPI · Pydantic · PostgreSQL 17 · SQLAlchemy · Alembic · MLflow (offline) · pytest · Ruff · GitHub Actions · Docker Compose (local PostgreSQL).

## Run locally

Requirements: Python 3.12, Docker Compose with `--wait` support, and the IEEE-CIS
source data for building the serving artifact.

~~~bash
git clone https://github.com/Sujeet-Kona/RiskForge.git
cd RiskForge
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
~~~

Review the root `.env` before sourcing it. Use a **disposable LOCAL PostgreSQL
server only**: match `DATABASE_URL` host, port, database, and role to the Compose
settings. Do not use a remote/shared server or an existing user database.
Compose reuses its named volume; if existing data is not explicitly disposable,
stop and provision an isolated local server instead.

Python and Alembic do **not** automatically load `.env`. Export it before any
migration, then wait for PostgreSQL readiness:

~~~bash
set -a
source .env
set +a
docker compose up -d --wait postgres
docker compose port postgres 5432
~~~

Stop on any error. Complete the
[read-only database target check](docs/reproducibility.md#local-api-and-database-checks)
and confirm the published port, database, and role belong to this disposable
local server. **Only then**, in the same exported environment, apply migrations:

~~~bash
.venv/bin/alembic upgrade head
.venv/bin/alembic check
~~~

Place the Kaggle source files under `data/raw/ieee_cis/` as described in
[dataset inputs](docs/reproducibility.md#dataset-inputs). The minimal serving
build needs `train_transaction.csv`; dataset inspection additionally needs
`train_identity.csv`. Data, generated behavioral features, and the fitted
serving artifact are not committed.

The minimal serving build has two commands:

~~~bash
PYTHONPATH=. python scripts/build_behavioral_features.py
PYTHONPATH=. python scripts/build_serving_artifact.py
~~~

The artifact builder fits the locked model on TRAIN and the isotonic calibrator
on VALIDATION. It does not require benchmark, calibration-evaluation, or policy
research reports. Dataset inspection is optional; separate research scripts
refit models and may overwrite reports. See
[optional research and report regeneration](docs/reproducibility.md#optional-research-and-report-regeneration)
before running them.

Start the API in the same exported environment:

~~~bash
uvicorn app.main:app --reload
~~~

Open http://127.0.0.1:8000/docs for the local API schema. The API does not train a replacement model if the artifact is missing or invalid.

## Repository map

- app/ — FastAPI schemas, lifecycle, feature construction, and scoring.
- frontend/ — React + TypeScript scoring dashboard and local API proxy.
- db/ and alembic/ — PostgreSQL models, decision audit persistence, review cases/events, and migrations.
- ml/ — data contract, temporal features, split, preprocessing, models, calibration, and artifacts.
- policy/ — decision thresholds and cost-assumption research.
- explain/ — TreeSHAP attribution and reason-code mapping.
- tracking/ — offline MLflow tracking.
- scripts/ — dataset, benchmark, calibration, policy, and parity workflows.
- tests/ — unit, API, database, and integration tests.
- reports/ and docs/ — saved evaluation evidence and methodology.

## Limitations

- Results use one chronological holdout from historical public data; they do not establish performance on current payment traffic.
- Bootstrap intervals resample transactions, not time blocks.
- The API trusts caller-supplied history and cannot independently verify its completeness, freshness, authenticity, or entity association.
- Prior-history rows are not stored with audit records, so the audit table alone cannot fully replay a score.
- Idempotency keys are global to this prototype and retained indefinitely; production use would scope keys to an authenticated tenant and define a retention policy.
- Scoring and read-only decision-history routes are unauthenticated and intended for local demonstration; only reviewer operations require the configured API key.
- The reviewer API key is a single local operator credential, not enterprise authentication or multi-user role-based access control.
- Policy costs and review-effectiveness assumptions are synthetic, not measured industry economics.
- TLS termination, rate limiting, production deployment, drift monitoring, and continuous retraining are outside the current implementation.

## Further documentation

- [API contract and examples](docs/api.md)
- [Interview guide](docs/interview-guide.md)
- [Verification record](docs/verification.md)
- [Model evaluation](docs/model-evaluation.md)
- [System design](docs/system-design.md)
- [Reproducibility](docs/reproducibility.md)
- [Temporal evaluation](docs/temporal_evaluation.md)
- [Policy provenance record](reports/policy_provenance.json)
