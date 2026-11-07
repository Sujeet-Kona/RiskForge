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
| Calibration | Isotonic regression selected using forward-chaining validation |
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
  end
  ARTIFACT -. loaded at startup .-> SCORE
  ARTIFACT -. calibrator .-> CALIBRATE
~~~

### Design boundaries

- Offline feature generation and online scoring use the same behavioral feature implementation.
- The API loads a frozen local artifact at startup; it does not train a fallback model during requests.
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

An additional expanding-window evaluation, paired moving-time-block bootstrap, and segmented calibration reliability reports are documented in [temporal evaluation](docs/temporal_evaluation.md). They require the IEEE-CIS source data. No additional robustness metrics are claimed until those runs complete.

## Decision policy

The serving path applies the frozen policy-001 thresholds to calibrated probability:

| Action | Calibrated probability |
|---|---|
| ALLOW | p < 0.075 |
| REVIEW | 0.075 ≤ p < 0.110 |
| BLOCK | p ≥ 0.110 |

Policy research used synthetic assumptions: missed-fraud cost 100 units, false-decline cost 10 units, manual-review cost 5 units, and 70% assumed fraud capture by review. These are experimental inputs, not measured payment-network economics. The policy is amount-blind and should not be interpreted as a validated real-world payment policy.

## Inference API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Service, artifact identity, and database readiness |
| POST | /v1/risk/score | Score one transaction using caller-supplied prior history |

The request includes the current transaction and a history array. History timestamps must be strictly earlier than the current transaction; same-timestamp and future records are rejected. Only the trailing 24 hours affect behavioral features. The API accepts at most 2,000 history records and rejects larger requests rather than truncating them. top_k controls returned reason signals (default 3).

The service loads a local artifact identified as xgb-001 / behavioral-v1. It contains preprocessing, XGBoost, the isotonic calibrator, and feature/model/policy metadata. A scoring request validates input, builds the 28-feature row, scores and calibrates it, applies policy-001, creates TreeSHAP and policy explanations, and persists the decision to PostgreSQL before returning success. Missing or incompatible artifacts fail closed; audit-write failure prevents a successful response.

TreeSHAP contributions are calculated on the raw XGBoost output margin (log-odds) and mapped back to the original 28 contract features. They are local model signals, not causal explanations, and are not calibrated-probability contributions. The policy explanation separately identifies the probability threshold band that produced the action.

## Verification

The recorded local offline-to-API parity check compared three real VALIDATION transactions: one ALLOW, one REVIEW, and one BLOCK. All 28 feature values matched between offline and API paths. Probabilities, decisions, reason codes, and policy explanations also matched. The largest absolute difference in the two floating-point amount-sum features was 1.310 × 10⁻¹⁰, within the 10⁻⁹ absolute tolerance.

This is a targeted three-case parity check, not exhaustive proof for all possible inputs or caller-supplied histories.

CI runs Ruff, Python compilation, Alembic migration/schema checks, and pytest with PostgreSQL 17. CI does not download IEEE-CIS, train models, build the serving artifact, run the real-data parity script, connect to an MLflow server, or deploy the API.

## Technology

Python 3.12 · pandas · NumPy · PyArrow · scikit-learn · XGBoost · SHAP/TreeSHAP · FastAPI · Pydantic · PostgreSQL 17 · SQLAlchemy · Alembic · MLflow (offline) · pytest · Ruff · GitHub Actions · Docker Compose (local PostgreSQL).

## Run locally

Requirements: Python 3.12, Docker Compose, and the IEEE-CIS source CSV files.

~~~bash
git clone https://github.com/Sujeet-Kona/RiskForge.git
cd RiskForge
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
docker compose up -d
.venv/bin/alembic upgrade head
~~~

Place the Kaggle competition CSV files under data/raw/ieee_cis/. The dataset, generated behavioral Parquet file, and serving artifact are not committed.

Build local features and the serving artifact:

~~~bash
PYTHONPATH=. python scripts/inspect_dataset.py
PYTHONPATH=. python scripts/build_behavioral_features.py
PYTHONPATH=. python scripts/evaluate_calibration.py
PYTHONPATH=. python scripts/select_decision_policy.py
PYTHONPATH=. python scripts/build_serving_artifact.py
~~~

Run the candidate benchmark separately:

~~~bash
PYTHONPATH=. python scripts/run_model_benchmark.py
~~~

Start the API:

~~~bash
uvicorn app.main:app --reload
~~~

Open http://127.0.0.1:8000/docs for the local API schema. The API does not train a replacement model if the artifact is missing or invalid.

## Repository map

- app/ — FastAPI schemas, lifecycle, feature construction, and scoring.
- db/ and alembic/ — PostgreSQL models, audit persistence, and migrations.
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
- Append-only audit behavior is an application convention, not database-enforced immutability; idempotency and duplicate suppression are not implemented.
- Policy costs and review-effectiveness assumptions are synthetic, not measured industry economics.
- Authentication, TLS termination, rate limiting, production deployment, drift monitoring, continuous retraining, and a human-review operation are outside the current implementation.

## Further documentation

- [Model evaluation](docs/model-evaluation.md)
- [System design](docs/system-design.md)
- [Reproducibility](docs/reproducibility.md)
- [Temporal evaluation](docs/temporal_evaluation.md)
