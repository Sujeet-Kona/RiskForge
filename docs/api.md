# Local API contract

RiskForge is a research prototype. Scoring and audit-history reads are
unauthenticated; reviewer operations require one server-configured local key.
Do not expose these routes publicly without a separate security design.
The running API's `/docs` and `/openapi.json` are the authoritative schemas.

## Readiness

```bash
curl --fail-with-body http://127.0.0.1:8000/health
```

A loaded artifact returns HTTP 200 with `status="ok"`, artifact metadata,
`model_version`, `feature_version`, `calibration_method`, `policy_version`, and
`database`. Scoring is ready only when `database="ready"` as well. Database
failure alone does not change the health HTTP status. A missing/invalid artifact
returns HTTP 503 with the same health shape and `status="unavailable"`, not the
application error envelope.

## Score one transaction

Run mutating examples only against an explicitly disposable local database.
This input is synthetic demonstration data, not a recorded customer payment.
The response is deliberately not hardcoded: use the real artifact's output.

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/risk/score \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: local-example-0001' \
  --data '{
    "transaction": {
      "TransactionID": "SYNTHETIC-EXAMPLE-1",
      "TransactionDT": 10000000,
      "TransactionAmt": 245.50,
      "card1": 13926,
      "ProductCD": "W",
      "card6": "debit"
    },
    "history": [],
    "top_k": 3
  }'
```

Required transaction fields: `TransactionID` (1–64 characters), `TransactionDT`
(nonnegative integer elapsed seconds), `TransactionAmt` (finite nonnegative
number), and `card1` (integer). Optional fields are `addr1`, `addr2`, the four
categorical inputs, and the C/D numerics listed in `app/schemas.py`. Unknown
fields are rejected. This is schema validation, not Pydantic global strict mode;
clients should send the documented JSON types rather than rely on coercion.

`history` defaults to `[]`, with at most 2,000 rows. Each row requires
`TransactionDT` and `TransactionAmt`; entity keys `card1`, `addr1`, `addr2` are
optional. Every row must be strictly earlier than the scored transaction,
including rows from other entities. Only matching entities within the trailing
24 hours affect features. Empty history is accepted but can mean missing context,
not necessarily that the card is genuinely new. The service cannot authenticate
or establish completeness of caller-supplied history.

`top_k` defaults to 3 and ranges from 1 to the feature-contract length (currently
28). The response includes calibrated `risk_probability`, policy `decision`,
risk-increasing `reason_codes`, `risk_decreasing_signals`, separate
`decision_reason_code`/`decision_explanation`, `model_detail`, `policy_detail`,
version identifiers, `feature_count`, and `history_transactions_used`.
TreeSHAP contributions are raw-margin model signals, not percentage-point
changes in calibrated probability and not causal evidence.

The immutable policy is ALLOW below 0.075, REVIEW from 0.075 up to 0.110, and
BLOCK at or above 0.110. Clients display the returned thresholds, not local
copies used to recompute decisions. See [policy provenance](../reports/policy_provenance.json).

### Retry semantics

`Idempotency-Key` is optional, 8–128 characters from letters, digits, `.`, `_`,
`:`, and `-`. Reuse the key for an unchanged request after a timeout. An identical
validated payload replays the stored response without another decision; a
changed payload with that key returns 409. Use a fresh key when changing input.
Keys are global and retained indefinitely in this single-operator prototype.
The decision, optional review case and response snapshot commit atomically.
Without a key, repeated scoring may create multiple audit records.

A failed audit write returns 503 without a score. Never interpret an error as an
ALLOW. A network timeout leaves the outcome unknown until retried with the same
key; absence of a response does not prove absence of a committed decision.

## Read decision history

```bash
curl --fail-with-body 'http://127.0.0.1:8000/v1/audit/decisions?limit=8&offset=0'
curl --fail-with-body 'http://127.0.0.1:8000/v1/audit/decisions?decision=REVIEW&limit=25&offset=0'
```

The newest-first page contains `items`, filtered `total`, `limit`, and `offset`.
`limit` defaults to 25 (1–100); `offset` defaults to 0 (nonnegative).
`decision`, when supplied, is ALLOW, REVIEW or BLOCK. Summary records include
transaction ID, probability, decision, model/feature/calibration/policy identity,
audit ID and timestamp. Audit history is not a feature store; prior history rows
are not retained, so these records alone cannot reproduce the full score.

## Human review

Configure `RISKFORGE_REVIEWER_API_KEY` and `RISKFORGE_REVIEWER_ID` on the backend.
The examples assume the key is already set in the shell environment; never put
it into a URL, Git, a `VITE_` variable or an example committed with a real value.
The header below exposes it to local process inspection while curl runs, so use
only a disposable development credential. The dashboard holds it in memory.

```bash
curl --fail-with-body 'http://127.0.0.1:8000/v1/reviews?status=PENDING&limit=25&offset=0' \
  -H "X-Reviewer-API-Key: ${RISKFORGE_REVIEWER_API_KEY:?Set a local reviewer key}"
```

Review pagination has the same bounds as decision history. `status` defaults to
PENDING and also accepts APPROVED or REJECTED to inspect completed cases.
Each case contains the original automated decision and explanation, scored
transaction inputs, version identifiers, human state and resolution metadata.
Only automated REVIEW decisions create cases.

After inspecting a disposable case, use its actual ID as `CASE_ID`:

```bash
curl --fail-with-body "http://127.0.0.1:8000/v1/reviews/${CASE_ID:?Set the inspected case ID}/resolve" \
  -H "X-Reviewer-API-Key: ${RISKFORGE_REVIEWER_API_KEY:?Set a local reviewer key}" \
  -H 'Content-Type: application/json' \
  --data '{"outcome":"APPROVED","comment":"Synthetic local workflow verification."}'
```

`outcome` must be APPROVED or REJECTED; `comment` is optional/null, maximum 2,000
characters. The response is `{"case": ...}`. One row lock and a one-way database
transition allow one terminal outcome and one append-only review event; the
original automated decision remains REVIEW. The server supplies reviewer
identity, rather than trusting a submitted reviewer name.

Resolution does **not** accept a replay/idempotency contract. Repeating it after
success returns 409; after a timeout, inspect terminal queues to learn the
recorded state and reviewer instead of assuming your retry succeeded. These
human labels are not independently verified fraud ground truth.

## Errors

Application errors use `{"error":{"code":"...","message":"..."}}` with
static client-facing messages. See [system design](system-design.md#error-responses)
for the code table. Important statuses:

- 401: reviewer key absent or incorrect.
- 404: missing route or review case; 405: wrong method.
- 409: idempotency payload conflict or already-resolved review case.
- 422: schema, history, header or query validation. FastAPI returns a `detail`
  array; explicit invalid idempotency-key characters return a `detail` string.
  The frontend extracts field/reason text rather than displaying echoed input.
- 503: artifact, audit store or reviewer configuration unavailable. Health uses
  its own shape; audit failure also covers history/queue reads and resolution.
- 500: inference/unhandled failure; no probability or decision is returned.

The API fails closed; it does not train, silently substitute a model, invent
history or return an unaudited decision. See [local setup](reproducibility.md)
and [verification evidence](verification.md) before running the examples.
