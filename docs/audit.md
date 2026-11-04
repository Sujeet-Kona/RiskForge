# RiskForge audit persistence

One PostgreSQL table, written once per successful decision, so that a decision can
be reconstructed later. This is an **audit store**: it is not part of model
training, it is not a feature store, and it is not distributed infrastructure.

---

## 1. What the database does

`POST /v1/risk/score` computes a decision, and then — after the decision is
complete and before the response is returned — writes one row describing it:

```
transaction → inference → decision/explanation → audit record → API response
```

The row is built from the response object the caller is about to receive, so what
is stored and what was reported are the same values by construction rather than by
convention. There is no second place where the persisted shape is defined.

The write cannot change the decision. `app/service.py` imports nothing from `db/`,
which is why `scripts/verify_serving_parity.py` still runs, and still proves
offline-versus-API agreement, with no database present.

---

## 2. What the database does NOT do

Stated plainly, because each of these would be a plausible-looking mistake:

- **It is not a source of behavioral history.** Nothing in the scoring path reads
  `risk_decisions`. Card velocity, address velocity, `card_has_history_1h` and
  every other behavioral feature are computed only from the `history` array in the
  request, under the locked point-in-time rule
  (`previous TransactionDT < current TransactionDT`). Reading prior audit rows back
  as context would break that rule and silently change what the model sees.
- **The prior transactions sent in `history` are not persisted.** Only
  `history_transactions_used`, their count, is. This is required by the rule above,
  not an oversight: a store that held them would be one query away from becoming a
  feature source.
- **It holds no labels.** There is no `is_fraud` column and no outcome column. This
  table cannot be used to retrain or to measure the model, and is not a
  ground-truth set.
- **It holds no customer identity.** No name, no email, no account number, no
  device fingerprint. The columns are the model-contract fields of the scored
  transaction and nothing invented beyond them. `card1` / `addr1` / `addr2` are
  IEEE-CIS entity keys, which are what the behavioral features are keyed on; they
  are recorded because a decision cannot be reconstructed without knowing what was
  scored.
- **It is not a queue, a cache or a workflow store.** No Redis, no Kafka, no
  Celery, no background worker, no retry. One synchronous write per successful
  inference is the whole persistence budget.
- **It is not production infrastructure.** No replication, no pooling proxy, no
  connection limit tuning, no TLS. Applying the committed migration in CI is a
  schema check, not a production rollout; see `docs/ci.md`.

The consequence of the second bullet is real and worth stating: an audit row tells
you what was scored and what was decided, but it is **not sufficient to recompute
the score from the database alone**. Reconstruction means the recorded
probability, decision, reason codes, explanation, versions and inputs — not a
replay.

---

## 3. PostgreSQL setup

Local development only needs a server on `localhost:5432` and a role that can
create databases (the test suite creates its own `riskforge_test` database).

With the container in this repository:

```bash
docker compose up -d
```

That is the entirety of `docker-compose.yml`: one `postgres:17` service, with its
data in a **named volume** so nothing a database writes can end up in the
repository. No application service, no Redis, no Kafka.

Or with a local install (Homebrew on macOS):

```bash
brew services start postgresql@17
createuser --createdb riskforge
psql -c "ALTER ROLE riskforge WITH PASSWORD 'riskforge'" -U postgres
createdb -O riskforge riskforge
```

Then create the schema:

```bash
.venv/bin/alembic upgrade head
```

`docker compose config` was used to validate the compose file. **The container
itself was not started during development** — the Docker daemon was unavailable on
the machine where this phase was built, and PostgreSQL was run locally instead.
The compose file is therefore reviewed and validated, not executed.

---

## 4. Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+psycopg://riskforge:riskforge@localhost:5432/riskforge` | The audit store. The only database RiskForge connects to. |
| `POSTGRES_USER` | `riskforge` | Container role |
| `POSTGRES_PASSWORD` | `riskforge` | Container password — **local development only** |
| `POSTGRES_DB` | `riskforge` | Container database |
| `POSTGRES_PORT` | `5432` | Published container port |

`DATABASE_URL` is read in exactly one place, `db/config.py`, and Alembic resolves
it through the same function — `alembic.ini` deliberately contains no
`sqlalchemy.url`, so the migration tool cannot be pointed at a different database
than the service, and no credential is committed.

A bare `postgres://` or `postgresql://` URL is normalized to
`postgresql+psycopg://`; anything that is not PostgreSQL is rejected at
configuration time, because the schema needs JSONB and `TIMESTAMPTZ`. The
rejection names the environment variable and the scheme and nothing else — a
connection string holds a password, so it is never echoed into an error.

The defaults above are development values. `.env.example` is the committed
template; `.env` is git-ignored and must never be committed.

---

## 5. Migrations

```bash
.venv/bin/alembic upgrade head        # create the schema
.venv/bin/alembic downgrade base      # remove it again
.venv/bin/alembic current             # where this database is
.venv/bin/alembic check               # does the model differ from the schema?
.venv/bin/alembic revision --autogenerate -m "..."   # after changing db/models.py
```

The schema is reproducible from the repository: it exists only as the initial
revision `7eccd8f63a8c`, and **the database is never edited by hand**. A generated
revision is reviewed before it is committed, and `alembic check` is run as a test
(`tests/test_db_migration.py`), so a column added to the model without a migration
fails the suite instead of quietly existing on one machine only.

`upgrade head → downgrade base → upgrade head` was verified against a real
PostgreSQL 17 database, and the reflected schema was compared with
`db/models.py` column by column.

---

## 6. Audit schema

One table, `risk_decisions`, append-only.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | `BIGINT` PK | Surrogate identity, `nextval` |
| `transaction_id` | `VARCHAR(64)` | The caller's identifier. **Indexed, not unique** |
| `risk_probability` | `DOUBLE PRECISION` | Full precision, `0.0`–`1.0` |
| `decision` | `VARCHAR(6)` | `ALLOW` / `REVIEW` / `BLOCK` |
| `reason_codes` | `JSONB` | Top-k risk-increasing model signals |
| `risk_decreasing_signals` | `JSONB` | Features that pushed risk down |
| `decision_reason_code` | `JSONB` | The single decision-rule code |
| `decision_explanation` | `TEXT` | The rendered explanation |
| `model_detail` | `JSONB` | Base value, margins, attribution totals |
| `policy_detail` | `JSONB` | The thresholds that were in force |
| `scored_inputs` | `JSONB` | The scored transaction, minus `TransactionID` |
| `top_k` | `INTEGER` | `>= 1`. Changes the explanation, so it is recorded |
| `model_version` | `VARCHAR(32)` | `xgb-001` |
| `feature_version` | `VARCHAR(32)` | `behavioral-v1` |
| `policy_version` | `VARCHAR(32)` | `policy-001` |
| `feature_count` | `INTEGER` | `> 0`. `28` |
| `history_transactions_used` | `INTEGER` | `>= 0` |
| `created_at` | `TIMESTAMPTZ` | `DEFAULT now()`, never supplied by the caller |

Indexes: `ix_risk_decisions_transaction_id`, `ix_risk_decisions_created_at`,
`ix_risk_decisions_decision_created_at (decision, created_at)`.

Check constraints, named from a convention so a migration and the model cannot
disagree about them:

- `ck_risk_decisions_risk_probability_in_unit_interval`
- `ck_risk_decisions_decision_is_a_policy_outcome`
- `ck_risk_decisions_feature_count_is_positive`
- `ck_risk_decisions_history_used_is_not_negative`
- `ck_risk_decisions_top_k_is_positive`

Two deliberate choices:

**JSONB, not normalized child tables.** A decision's explanation is a payload that
is read back whole. Splitting reason codes into their own table would mean a join
to reconstruct one audit record and a second place for the reason-code shape to
drift, with no query in this phase that needs it. JSONB still allows filtering
(`reason_codes @> '[{"feature": "C11"}]'`), which is the querying an audit needs.

**`transaction_id` is not unique.** It is caller-supplied, and scoring the same
transaction twice is two decisions, both of which happened. A unique constraint
would turn a legitimate second call into a write failure, and a write failure
means a refused response — the wrong outcome for an append-only log.

Validation happens at both boundaries, because the database alone cannot be relied
on to validate application input: `db.repository.AuditRecord` rejects a bad
probability, a decision outside the policy vocabulary, a blank version identifier
or a malformed payload **in Python, before a statement is issued**, and the CHECK
constraints reject the same things if that layer is bypassed. Both paths are
tested (`tests/test_db_models.py`, `tests/test_db_repository.py`).

---

## 7. API persistence behaviour

The `POST /v1/risk/score` response contract is unchanged: no field added, no field
renamed, no field removed. Persistence is invisible to a caller whose request
succeeds.

A rejected request (`422`) records nothing, because no decision was produced.

`GET /health` gains one field, `database`: `ready` or `unavailable`. `status`
keeps its previous meaning — it describes the model artifact — so the endpoint
still returns `200` and `status: ok` when the audit store is down. Readiness is one
`SELECT 1`; it touches no table.

---

## 8. Database failure behaviour

**A decision that cannot be recorded is not returned.**

If the audit write fails, the client receives:

```
503
{"error": {"code": "audit_unavailable", "message": "The risk decision was produced but could not be recorded in the audit store, so it is not being returned. RiskForge does not report a decision it cannot audit. Nothing about the decision is included in this response; retry once the audit store is reachable."}}
```

Why not return the score anyway: the stated purpose of the system includes
auditability, and a `200` would assert that this decision is in the audit trail
when it is not. That is the one false claim this design refuses to make. The
probability and the decision are withheld entirely — a partial success response
would be read as a success.

Why `503` and not `500`: `500 inference_failed` already means "scoring broke".
`503 audit_unavailable` means "scoring worked, recording did not", which is a
different operational event with a different remedy.

The response is sanitized. It carries no connection string, no SQL, no driver or
exception name, no host or port, no filesystem path and no traceback. The real
driver error is logged server-side at `ERROR` with its traceback, chained as the
exception's cause, so an operator can diagnose it and a client cannot.
`tests/test_app_audit_api.py` asserts the absence of each of those fragments,
including against a genuine psycopg connection failure rather than a test double.

The failure is per-request. Nothing is latched and no restart is needed: the next
request succeeds as soon as the store is reachable.

---

## 9. Layout

| Path | Role |
| --- | --- |
| `db/config.py` | The only place `DATABASE_URL` is read and validated |
| `db/session.py` | Engine, session factory, `is_ready()`, `dispose()` |
| `db/models.py` | The `risk_decisions` declarative model and its naming convention |
| `db/repository.py` | `AuditRecord` validation, `AuditStore` protocol, `AuditRepository` |
| `app/audit.py` | Response → `AuditRecord`, and `AuditUnavailableError` |
| `app/dependencies.py` | Builds the store once at startup; the only module in `app/` that reaches the engine |
| `alembic/` | The schema |
| `docker-compose.yml` | One local PostgreSQL |

`db/` imports `policy.decision` (for the decision vocabulary), SQLAlchemy and the
standard library. It imports nothing from `app/`, `ml/` or `explain/`, and contains
no model logic: the repository does not score, does not calculate risk, does not
apply a threshold and does not know what a model is. No module in `app/` imports
`sqlalchemy`, `psycopg` or `alembic`. These boundaries are asserted by
`tests/test_serving_discipline.py`, so they cannot erode quietly.

---

## 10. Tests

| File | Needs PostgreSQL | Covers |
| --- | --- | --- |
| `tests/test_db_config.py` | no | URL resolution, normalization, rejection, no credential leakage |
| `tests/test_db_models.py` | no | Columns, types, nullability, indexes, constraints, record validation |
| `tests/test_db_repository.py` | yes | Round trip, precision, JSONB, timestamps, constraint enforcement, failure sanitization |
| `tests/test_db_migration.py` | yes | Head applied, `alembic check` clean, reflected schema, downgrade/upgrade |
| `tests/test_app_audit_api.py` | both halves | What is recorded, response contract, refusal path, `/health` readiness |

Database tests create `riskforge_test` themselves and run Alembic
`downgrade base → upgrade head` in the session fixture, so the schema they assert
against was produced in that run rather than left over. When no PostgreSQL is
reachable locally they **skip, and the skip is reported**. CI sets
`RISKFORGE_REQUIRE_POSTGRES` so the same fixture fails instead of skipping —
the suite cannot go green without a database. Every other test uses an
in-memory audit double, so the wire contract does not depend on a server being
up.

---

## 11. Known limitations

- **A reconstructed decision cannot be replayed.** The prior transactions that
  produced the behavioral features are not stored (§2). Replaying one would need
  them, and storing them is what the phase forbids.
- **One synchronous write per request, on the request path.** A slow or
  unreachable database shows up directly in request latency, and a refused
  decision is a lost decision — there is no outbox and no retry. That is an
  accepted trade for this phase, not a scalable design.
- **No retention, partitioning or archival policy.** The table only grows.
- **No read API.** Rows are queried with SQL. Exposing them over HTTP would need
  authorization, which is explicitly out of scope here.
- **`docker-compose.yml` was validated with `docker compose config` but not
  executed**, because no Docker daemon was available during development.
