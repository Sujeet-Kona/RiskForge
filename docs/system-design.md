# System design

## Inference and decision lifecycle

The online path is intentionally separate from model development:

1. FastAPI validates the transaction and caller-supplied strictly-prior history.
2. The feature layer constructs the 28-feature model input.
3. The locked serving artifact produces a raw score and calibrated probability.
4. The artifact's `model_version`, `feature_version`, and `calibration_method`
   are carried into the response and audit record.
5. `policy-001` maps the calibrated probability to `ALLOW`, `REVIEW`, or
   `BLOCK`.
6. The explanation layer returns TreeSHAP model signals and a separate policy
   explanation.
7. PostgreSQL records the decision before the API returns success. A `REVIEW`
   decision also creates a pending review case in the same transaction. If the
   request supplies an `Idempotency-Key`, its request fingerprint and original
   response snapshot are committed in that transaction too.

The API loads the artifact at startup. It does not fit preprocessing, train a
model, calibrate probabilities, or search policy thresholds during a request.

## API

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Reports service, artifact identity, calibration method, and database readiness. |
| `POST /v1/risk/score` | Scores one transaction using supplied prior history. |
| `GET /v1/audit/decisions` | Returns a bounded, newest-first decision history; supports an optional decision filter. |
| `GET /v1/reviews` | Returns the authenticated pending review queue. |
| `POST /v1/reviews/{case_id}/resolve` | Records a one-time human review outcome. |

Interactive request/response schemas are available at `/docs` when the local API
is running.

A score request contains a transaction, a `history` array, and optional
`top_k` for the number of reason codes. History must contain only records
strictly earlier than the transaction being scored. Records sharing the current
transaction timestamp must not be included.

The service is stateless with respect to transaction history. It does not query
the audit table to construct features. The caller is responsible for supplying
history for the relevant entities.

## Idempotency

The scoring endpoint accepts an optional `Idempotency-Key` header. The API hashes
the canonical validated request body. Repeating the same key with the same
request returns the original stored response without creating a second decision.
Reusing the key with a different request returns HTTP 409.

The unique key constraint handles concurrent retries: one transaction wins and
the losing transaction rolls back its decision and review-case writes before
reading the winner's response. The decision, optional review case, and response
snapshot are committed atomically.

Keys are global to this single-operator prototype and retained indefinitely.
Production use would scope keys to an authenticated tenant and define a retention
policy.

## Human review

Only automated `REVIEW` outcomes create review cases. A reviewer can inspect the
original transaction details, calibrated risk, TreeSHAP reason signals, policy
explanation, and model/feature/calibration/policy identifiers.

A configured reviewer key is required for both queue reads and resolution
actions. The current demo uses one operator identity configured by
`RISKFORGE_REVIEWER_ID` and one secret configured by
`RISKFORGE_REVIEWER_API_KEY`. This is local demonstration authentication, not a
production identity provider or multi-user role system.

Review outcomes are `APPROVED` or `REJECTED`. The original automated decision
remains `REVIEW`; human resolution is stored separately. Each case can move
once from `PENDING` to a terminal state.

## Model explanations and policy explanations

The explanation layer uses TreeSHAP contributions from the raw XGBoost model
output (log-odds) to rank feature signals. A controlled reason-code taxonomy
groups those signals into readable categories.

The policy explanation is separate: it states which threshold band contains the
calibrated probability. A SHAP contribution is not itself a decision rule, and
the reason-code ranking does not explain business costs.

Explanations are local to an individual prediction. They do not establish causal
effects or global feature importance.

## PostgreSQL decision records

A successful score is written to PostgreSQL before the response is returned.
The record captures the scored transaction and decision details, model version,
feature-contract version, calibration method, policy version, and explanation
output. `GET /v1/audit/decisions` exposes a bounded, paginated view for the UI.

Database protections:

- `risk_decisions` and `risk_review_events` reject UPDATE and DELETE through
  PostgreSQL triggers.
- `risk_review_cases` permits only a one-way `PENDING` to `APPROVED` or
  `REJECTED` transition; resolution metadata is constrained by the database.
- `risk_idempotency_records` are append-only and hold the request fingerprint
  and response snapshot.
- These controls protect against ordinary application mistakes, not a database
  administrator who can alter the schema or disable triggers.

Important boundaries:

- Prior-history rows are not stored, so an audit row alone cannot reproduce the
  full feature vector or replay the original score.
- Scoring and read-only decision-history routes are unauthenticated and intended
  for the local demo. Reviewer operations require the configured local key.
- The write is synchronous on the request path. A database failure prevents a
  successful scoring response.

The audit table is not a feature store.

## Experiment tracking

MLflow is used by the offline tracking module to record experiment parameters,
metrics, feature-contract metadata, data identity, environment information,
source revision, and model artifacts.

MLflow is not part of the inference path. The FastAPI service does not contact an
MLflow tracking server or model registry to score a request; it loads the
separate serving artifact built by `scripts/build_serving_artifact.py`.

## Main code areas

| Path | Responsibility |
| --- | --- |
| `ml/features/` and `ml/split/` | Point-in-time features and chronological splitting |
| `ml/models/` | Preprocessing, estimators, calibration, and artifact handling |
| `policy/` | Decision thresholds and policy evaluation |
| `explain/` | TreeSHAP attribution and reason-code mapping |
| `app/` | FastAPI schemas, lifecycle, inference, audit, and reviewer operations |
| `db/` and `alembic/` | PostgreSQL decisions, review cases/events, idempotency, and migrations |
| `tracking/` | Offline MLflow experiment tracking |
| `scripts/` | Data, model, evaluation, artifact, and parity workflows |
| `frontend/` | React + TypeScript scoring dashboard and review queue |
| `tests/` | Unit, API, database, and integration tests |
