# RiskForge inference API

A FastAPI service that scores one transaction with the locked XGBoost model, the
locked isotonic calibration, the frozen `policy-001` decision rule and the
explanation layer — and records every decision it returns in a PostgreSQL audit
store.

**The service keeps no transaction history of its own.** Behavioral features are
computed from the `history` array the caller sends with each request. The audit
store records decisions that have already been made and is never read back as
feature history — see [docs/audit.md](audit.md).

---

## 1. What this is, and what it is not

It is an inference API. Given one transaction and the caller-supplied history for
that transaction's entities, it returns a calibrated fraud-risk probability, an
`ALLOW` / `REVIEW` / `BLOCK` decision, and the reason codes behind both.

It is not:

- **not real-time fraud detection.** The service holds no transaction history. It
  can only see what the caller sends in the request, so it cannot detect a burst
  of activity across requests, and it cannot compute `card_txn_count_1h` for a
  card it has never been told about.
- **not a production deployment.** There is no TLS, no auth, no rate limiting, no
  monitoring, no horizontal scaling story. The only container in the repository
  is a local PostgreSQL for development. Those are later phases.
- **not a feature store.** The audit store is append-only and write-only from the
  API's point of view: nothing in the scoring path reads it.

Everything it reports is a deterministic function of the frozen artifact and the
request. There is no randomness, no LLM and no external call other than the one
audit write.

---

## 2. Building the serving artifact

The API never trains. It loads one bundle, built offline:

```bash
PYTHONPATH=. python scripts/build_serving_artifact.py
```

That script fits the locked XGBoost configuration on TRAIN, fits isotonic
calibration on VALIDATION, and writes `artifacts/serving/riskforge_serving_artifact.joblib`.
It refuses to publish if `scale_pos_weight` is not the locked `27.410063`, and it
never reads the TEST partition.

The bundle contains the fitted preprocessing + XGBoost pipeline, the fitted
calibrator, the frozen policy identity, the feature contract and its version, the
provenance (row counts, build timestamp) and the `scikit-learn` / `xgboost`
versions it was built under. Loading it under different library versions is
refused, because a pickled estimator can unpickle successfully and still score
differently.

The artifact directory is gitignored. Without it the service starts and returns
`503` — it does not train a substitute.

---

## 3. Starting the server

```bash
docker compose up -d          # local PostgreSQL for the audit store
.venv/bin/alembic upgrade head
uvicorn app.main:app --reload
```

The artifact is loaded once, during application startup. It is not re-read per
request, and nothing is fitted at request time. The database engine is also built
once at startup, but opening it connects to nothing: an unreachable database is
discovered when a decision is written, not when the service boots.

Interactive OpenAPI docs are served by FastAPI at `/docs`.

---

## 4. `GET /health`

Liveness, artifact identity, and audit-store readiness. It performs no inference.

`200` when the artifact loaded:

```json
{
  "status": "ok",
  "model_version": "xgb-001",
  "feature_version": "behavioral-v1",
  "policy_version": "policy-001",
  "database": "ready",
  "artifact": {
    "artifact_schema_version": 1,
    "model_version": "xgb-001",
    "feature_version": "behavioral-v1",
    "policy_version": "policy-001",
    "calibration_method": "isotonic",
    "review_threshold": 0.075,
    "block_threshold": 0.11,
    "scale_pos_weight": 27.410063065533315,
    "n_transformed_columns": 162,
    "n_features": 28,
    "n_train_rows": 414446,
    "n_calibration_rows": 88013,
    "library_versions": {"scikit-learn": "1.9.1", "xgboost": "3.4.1"},
    "built_at_utc": "2026-11-02T17:55:55+00:00"
  }
}
```

`503` when it did not:

```json
{
  "status": "unavailable",
  "model_version": null,
  "feature_version": null,
  "policy_version": null,
  "artifact": null,
  "database": "ready"
}
```

`status` describes the model artifact and keeps its previous meaning, so an
existing caller sees no change. `database` is reported separately as `ready` or
`unavailable`, because the two can disagree: a service whose artifact is fine but
whose audit store is down is alive and still `200`, but it cannot return a
decision. Readiness is one `SELECT 1` round trip — no table access, no query plan
— so `/health` stays cheap enough to poll.

---

## 5. `POST /v1/risk/score`

### 5.1 Request

```json
{
  "transaction": { "...": "..." },
  "history": [ { "...": "..." } ],
  "top_k": 3
}
```

Field names in the request are the locked contract names verbatim
(`TransactionDT`, `ProductCD`, `C13`). A translation table between the wire
format and `MODEL_FEATURES` would be a second place for the feature contract to
be expressed, and therefore a second place for it to drift. Unknown fields are
rejected, so a caller who sends `product_cd` gets a `422` rather than a silently
imputed gap.

**`transaction` — required**

| Field | Type | Rule |
| --- | --- | --- |
| `TransactionID` | string | required, non-empty. Echoed back, never used as a feature. |
| `TransactionDT` | integer | required, `>= 0`. Elapsed seconds, as in IEEE-CIS — not a wall-clock date. |
| `TransactionAmt` | float | required, `>= 0`, finite. |
| `card1` | integer | required. Card entity key for the behavioral history. |
| `addr1`, `addr2` | float or null | optional, finite. Together they form the address entity key. |
| `ProductCD`, `card6`, `P_emaildomain`, `R_emaildomain` | string or null | optional. Unseen levels are valid. |
| `C1`, `C2`, `C11`, `C13` | float or null | optional, finite. |
| `D2`, `D3`, `D4`, `D8`, `D10`, `D11`, `D15` | float or null | optional, finite. |

That is the whole 28-feature contract and nothing else. `TransactionID` and
`TransactionDT` are request fields but not model features; the rest of the
IEEE-CIS columns are not part of the locked contract and are not accepted.

A missing categorical or C/D value is not an error. Gaps are signal in this
dataset: the preprocessor imputes them and raises a missing indicator.

**`history` — optional, defaults to `[]`**

Each entry carries only the five columns the behavioral features actually read:

| Field | Type | Rule |
| --- | --- | --- |
| `TransactionDT` | integer | required, `>= 0`, and **strictly less than** `transaction.TransactionDT` |
| `TransactionAmt` | float | required, `>= 0`, finite |
| `card1` | integer or null | optional |
| `addr1`, `addr2` | float or null | optional |

No categorical or C/D value of a *historical* transaction contributes to any
feature of the current one, so the schema does not ask for them. Accepting fields
that cannot matter would imply that they do.

**`top_k` — optional, default `3`**, between `1` and `28`.

### 5.2 The point-in-time rule

The locked feature implementation may only read rows with
`TransactionDT < current TransactionDT`. The API enforces that on the request:

- a history entry at the current timestamp is rejected;
- a history entry after it is rejected;
- one invalid entry fails the whole request, naming its position.

Invalid history is **never silently filtered**. Dropping a future row would let a
caller believe their request was scored as sent.

The windows are open at both ends. A row exactly 3,600 seconds before the current
transaction is outside the 1h window and inside the 24h window; a row exactly
86,400 seconds before it is outside both.

### 5.3 How much history to send

Send the trailing 24 hours of the transaction's own entities: every prior
transaction matching `card1`, plus every prior transaction matching the
(`addr1`, `addr2`) pair. Anything older than 24 hours cannot affect any feature.

The bound is measured, not guessed. Over all 590,540 real IEEE-CIS rows the most
any single transaction needs is `card_txn_count_24h + address_txn_count_24h = 1864`
rows. `MAX_HISTORY_TRANSACTIONS` is therefore `2000`, and a request above it is
**rejected, not truncated** — trimming history would quietly change the behavioral
features and produce a score that no longer matches the offline pipeline.

### 5.4 Response

`200`, with the probability, the decision, and the two halves of the explanation
kept separate:

```json
{
  "transaction_id": "3401505",
  "risk_probability": 0.04310344827586207,
  "decision": "ALLOW",
  "reason_codes": [
    {
      "code": "MODEL_SIGNAL_C11",
      "title": "Model signal from C11",
      "type": "model_signal",
      "description": "C11 is a numeric benchmark feature from the IEEE-CIS dataset with undocumented upstream semantics. Its imputed value and missing indicator contributed to the model's fraud-risk score. No business meaning is claimed for it.",
      "feature": "C11",
      "contribution": 0.5715078711509705,
      "direction": "risk_increasing",
      "n_transformed_columns": 1
    },
    {
      "code": "MODEL_SIGNAL_C13",
      "title": "Model signal from C13",
      "type": "model_signal",
      "description": "C13 is a numeric benchmark feature from the IEEE-CIS dataset with undocumented upstream semantics. Its imputed value and missing indicator contributed to the model's fraud-risk score. No business meaning is claimed for it.",
      "feature": "C13",
      "contribution": 0.3246549069881439,
      "direction": "risk_increasing",
      "n_transformed_columns": 1
    },
    {
      "code": "ADDRESS_TRANSACTION_VELOCITY_SIGNAL",
      "title": "Recent address transaction activity",
      "type": "model_signal",
      "description": "The count of prior transactions observed for this address in the point-in-time 1h and 24h windows contributed to the model's fraud-risk score.",
      "feature": "address_txn_count_24h",
      "contribution": 0.16099920868873596,
      "direction": "risk_increasing",
      "n_transformed_columns": 1
    }
  ],
  "risk_decreasing_signals": [
    {
      "code": "ADDRESS_AMOUNT_VOLUME_SIGNAL",
      "title": "Prior address spend volume",
      "type": "model_signal",
      "description": "The summed transaction amount observed for this address in the preceding 24h window contributed to the model's fraud-risk score.",
      "feature": "address_amount_sum_24h",
      "contribution": -0.3132265508174896,
      "direction": "risk_decreasing",
      "n_transformed_columns": 1
    },
    {
      "code": "MODEL_SIGNAL_C1",
      "title": "Model signal from C1",
      "type": "model_signal",
      "description": "C1 is a numeric benchmark feature from the IEEE-CIS dataset with undocumented upstream semantics. Its imputed value and missing indicator contributed to the model's fraud-risk score. No business meaning is claimed for it.",
      "feature": "C1",
      "contribution": -0.26017844676971436,
      "direction": "risk_decreasing",
      "n_transformed_columns": 1
    },
    {
      "code": "MODEL_SIGNAL_C2",
      "title": "Model signal from C2",
      "type": "model_signal",
      "description": "C2 is a numeric benchmark feature from the IEEE-CIS dataset with undocumented upstream semantics. Its imputed value and missing indicator contributed to the model's fraud-risk score. No business meaning is claimed for it.",
      "feature": "C2",
      "contribution": -0.2599486708641052,
      "direction": "risk_decreasing",
      "n_transformed_columns": 1
    }
  ],
  "decision_reason_code": {
    "code": "POLICY_BELOW_REVIEW_THRESHOLD",
    "title": "Risk below the review threshold",
    "type": "decision_rule",
    "description": "The calibrated risk probability fell below the frozen policy's REVIEW threshold, so the policy returned ALLOW."
  },
  "decision_explanation": "Risk probability 0.043103 is below the REVIEW threshold 0.075000.",
  "model_detail": {
    "base_value": 0.03433862328529358,
    "raw_margin": -0.3683631265305394,
    "raw_model_probability": 0.40893658995628357,
    "attribution_space": "raw_margin_log_odds",
    "attribution_space_note": "Contributions are expressed in the XGBoost raw margin (log-odds) space, not in probability space. The base value plus the sum of every contribution equals the model's raw margin, and the raw model probability is sigmoid(margin). Because the locked isotonic calibration is monotonic, the sign of a contribution still tells you whether that feature pushed the deployed risk probability up or down; the magnitude does not carry over, since calibration rescales the margin non-linearly. Reason codes therefore describe model-level risk signals, not the calibration mapping.",
    "total_risk_increasing": 1.4653008726743337,
    "total_risk_decreasing": -1.8680026224901667
  },
  "policy_detail": {"review_threshold": 0.075, "block_threshold": 0.11},
  "model_version": "xgb-001",
  "feature_version": "behavioral-v1",
  "policy_version": "policy-001",
  "feature_count": 28,
  "history_transactions_used": 2
}
```

Reading it:

- `reason_codes` are **model signals** — TreeSHAP contributions in the raw
  log-odds space, largest first, capped at `top_k`. They explain what moved the
  model's score. They do not explain the band.
- `decision_reason_code` and `decision_explanation` are the **decision rule**. The
  two are never merged: the strongest model signal is not why a transaction was
  blocked.
- `risk_decreasing_signals` are reported for completeness and are never fraud
  evidence.
- A reason code can appear more than once. The taxonomy has 24 codes for 28
  features, so four sibling pairs share one code (`card_txn_count_1h`/`24h`,
  `address_txn_count_1h`/`24h`, `card_has_history_1h`/`24h`,
  `TransactionAmt`/`amount_log1p`). One entry per contributing feature is the
  faithful Phase 5 behaviour, so the `feature` field, not the `code`, is what
  distinguishes them.
- `risk_probability` is the **full-precision** calibrated value the policy
  consumed. It is not rounded before the decision is taken; rounding for display
  is the caller's business.

### 5.5 Version fields

`xgb-001`, `behavioral-v1` and `policy-001` are **project artifact identifiers,
not registry versions**. RiskForge has no model registry, so there is
nothing behind them except the local bundle and the locked feature contract.
`policy-001` is the frozen Phase 4 policy and is the one identifier with a formal
definition behind it: `T_REVIEW = 0.075`, `T_BLOCK = 0.110`.

---

## 6. Errors

Error bodies are always `{"error": {"code": ..., "message": ...}}`, except FastAPI's
own `422` validation detail. They never carry a stack trace, a filesystem path or
an exception class name; the underlying detail is logged server-side.

| Status | When | Code |
| --- | --- | --- |
| `422` | Malformed body, missing required field, negative or non-finite amount, negative `TransactionDT`, unknown field, `top_k` out of range, history at or after the current timestamp, history above the bound | FastAPI validation detail |
| `500` | An unexpected failure during scoring | `inference_failed` |
| `503` | The artifact is missing | `artifact_not_found` |
| `503` | The artifact exists but is rejected (wrong contract, wrong width, unfitted calibrator, foreign library versions) | `artifact_invalid` |
| `503` | A decision was produced but could not be written to the audit store | `audit_unavailable` |

A failure is never reported as a score. The `500` body contains no
`risk_probability` and no `decision`, because inventing one would be worse than
failing. The `audit_unavailable` body is held to the same rule for a different
reason: the decision was computed, but returning it would claim an audit trail
that does not exist.

`audit_unavailable` is a `503` rather than a `500` so that "we could not record
it" is distinguishable from "scoring broke". It is per-request: nothing is
latched, and the next request succeeds once the store is reachable. The driver
error is logged server-side with its traceback and never reaches the client — no
DSN, no SQL, no driver name, no filesystem path.

---

## 7. Where the decision is written

After the decision is complete and before the response is returned, the endpoint
writes one audit row built from the response object itself, so what is stored and
what was reported are the same values by construction rather than by convention:

```
transaction → inference → decision/explanation → audit record → API response
```

The write cannot change the decision. `app/service.py` imports nothing from `db/`,
so the scoring path is identical to the one that produced the offline metrics, and
`scripts/verify_serving_parity.py` still runs with no database present.

The caller still owns the history. A deployment that sends no history gets a score
computed from the current transaction alone, with every behavioral count at zero,
and that score will differ from the offline pipeline's for the same row. The audit
store does not fix that, and must not be used to: it is append-only, and reading
prior decisions back as behavioral context would break the point-in-time contract.
That is not a limitation of this phase, it is the rule the phase was built under.

See [docs/audit.md](audit.md) for the schema, the migration commands and the full
failure behaviour.

---

## 8. One inference path

There is no serving-side feature logic. `app/features.py` assembles a frame from
the request and calls `ml.features.behavioral.build_behavioral_features` — the
same function that produced the offline training artifact — then selects
`MODEL_FEATURES`. Scoring then calls `predict_fraud_probability`,
`apply_calibrator` and `explain_transaction`, the same functions the offline
scripts call.

Two differences between the serving frame and the offline frame are known,
documented and measured rather than assumed away:

- `card1` is `int64` offline and `float64` online, because a history row may omit
  it. The behavioral features are counts and sums keyed on equality, and integer
  and float keys of the same value group identically.
- Categorical columns are built with explicit offline dtypes, because pandas 3.x
  infers `object` for a column whose only value is `None`, where the offline frame
  carries `str`.

`scripts/verify_serving_parity.py` scores real VALIDATION transactions through both
routes and requires them to agree. On the run that produced this document it
reported identical calibrated probabilities, decisions, reason codes, explanations
and version fields for one ALLOW, one REVIEW and one BLOCK transaction; all 28
features matched, with a worst-case delta of `1.310e-10` on `card_amount_sum_24h`,
which is a summation-order difference between a 590,540-row replay and a window
sum and has no effect on the score.

```bash
PYTHONPATH=. python scripts/verify_serving_parity.py
```

---

## 9. Example

```bash
curl -s -X POST http://127.0.0.1:8000/v1/risk/score \
  -H 'Content-Type: application/json' \
  -d '{
    "transaction": {
      "TransactionID": "3401505",
      "TransactionDT": 12766932,
      "TransactionAmt": 61.0,
      "card1": 13926,
      "addr1": 299.0,
      "addr2": 87.0,
      "ProductCD": "W",
      "card6": "debit",
      "P_emaildomain": "gmail.com",
      "R_emaildomain": null,
      "C1": 1.0, "C2": 1.0, "C11": 5.0, "C13": 1.0,
      "D2": null, "D3": null, "D4": null, "D8": null,
      "D10": 1.0, "D11": null, "D15": null
    },
    "history": [
      {"TransactionDT": 12760000, "TransactionAmt": 45.0, "card1": 13926, "addr1": 299.0, "addr2": 87.0},
      {"TransactionDT": 12700000, "TransactionAmt": 90.0, "card1": 13926, "addr1": 299.0, "addr2": 87.0}
    ],
    "top_k": 3
  }'
```

That request produced the response in §5.4.

---

## 10. Layout

| Path | Role |
| --- | --- |
| `app/config.py` | Artifact path, default `top_k`, history bound, history note |
| `app/schemas.py` | Pydantic request and response models; the wire contract |
| `app/features.py` | Request payload to the locked 28-feature frame, via the offline feature function |
| `app/service.py` | Features to probability to decision to explanation |
| `app/audit.py` | Response object to audit record, and the refusal when it cannot be stored |
| `app/dependencies.py` | Startup artifact and audit-store loading, and the per-request dependencies |
| `app/main.py` | Application factory, routes, error handlers |
| `db/config.py` | The only place `DATABASE_URL` is read |
| `db/session.py` | Engine, session factory, readiness probe |
| `db/models.py` | The `risk_decisions` declarative model |
| `db/repository.py` | `create_audit_record` / `get_audit_record`, and the record's validation |
| `alembic/` | The schema, reproducible from the repository |
| `ml/models/artifact.py` | Artifact bundle, validation, save and load |
| `policy/locked.py` | The one frozen `policy-001` definition |
| `scripts/build_serving_artifact.py` | The only producer of the bundle |
| `scripts/verify_serving_parity.py` | Offline-versus-API regression on real data |

Only `app/dependencies.py` and `app/main.py` import FastAPI. No module in `app/`
imports `sqlalchemy`, `psycopg` or `alembic`: the database is reached through
`db.repository`'s `AuditStore` boundary and nowhere else. `policy/` imports
nothing from `ml/`, `explain/`, `app/` or `db/`, and `db/` imports nothing from
`app/`, `ml/` or `explain/` — only `policy.decision`, for the decision vocabulary.
