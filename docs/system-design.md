# System design

## Inference flow

The online path is intentionally separate from model development:

1. FastAPI validates the transaction and caller-supplied history.
2. The feature layer constructs the 28-feature model input.
3. The locked serving artifact produces a raw score and calibrated probability.
4. `policy-001` maps the probability to `ALLOW`, `REVIEW`, or `BLOCK`.
5. The explanation layer returns model-signal reason codes and a separate decision explanation.
6. The API writes the decision record to PostgreSQL before returning success.

The API loads the artifact at startup. It does not fit preprocessing, train a model, calibrate probabilities, or search policy thresholds during a request.

## API

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Reports service, artifact identity, and database readiness. |
| `POST /v1/risk/score` | Scores one transaction using the supplied prior history. |

Interactive request/response schemas are available at `/docs` when the local API is running.

A score request contains a transaction, a `history` array, and optional `top_k` for the number of reason codes. History must contain only records strictly earlier than the transaction being scored. Records sharing the current transaction timestamp must not be included.

The service is stateless with respect to transaction history. It does not query the audit table to construct features. The caller is responsible for supplying history for the relevant entities.

## Model explanations and policy explanations

The explanation layer uses TreeSHAP contributions from the raw XGBoost model output (log-odds) to rank feature signals. A controlled reason-code taxonomy groups those signals into readable categories.

The policy explanation is separate: it states which threshold band contains the calibrated probability. A SHAP contribution is not itself a decision rule, and the reason-code ranking does not explain business costs.

Explanations are local to an individual prediction. They do not establish causal effects or global feature importance.

## PostgreSQL decision records

A successful score is written to PostgreSQL before the response is returned. The record captures the scored transaction and decision details, including model/policy identifiers and explanation output.

Important boundaries:

- Prior-history rows are not stored, so an audit row alone cannot reproduce the full feature vector or replay the original score.
- Append-only behavior is enforced by application behavior; database permissions or triggers do not make the table immutable.
- The write is synchronous on the request path. A database failure prevents a successful scoring response.
- Request idempotency, duplicate suppression, retention, and archival policies are not implemented.

The audit table is not a feature store.

## Experiment tracking

MLflow is used by the offline tracking module to record experiment parameters, metrics, feature-contract metadata, data identity, environment information, source revision, and model artifacts.

MLflow is not part of the inference path. The FastAPI service does not contact an MLflow tracking server or model registry to score a request; it loads the separate serving artifact built by `scripts/build_serving_artifact.py`.

## Main code areas

| Path | Responsibility |
| --- | --- |
| `ml/features/` and `ml/split/` | Point-in-time features and chronological splitting |
| `ml/models/` | Preprocessing, estimators, calibration, and artifact handling |
| `policy/` | Decision thresholds and policy evaluation |
| `explain/` | TreeSHAP attribution and reason-code mapping |
| `app/` | FastAPI schemas, lifecycle, and inference orchestration |
| `db/` and `alembic/` | PostgreSQL audit repository and migrations |
| `tracking/` | Offline MLflow experiment tracking |
| `scripts/` | Data, model, evaluation, artifact, and parity workflows |
| `tests/` | Unit, API, database, and integration tests |
