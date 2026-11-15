# RiskForge interview guide

**Scope:** an offline research and local decisioning prototype using public IEEE-CIS data, not a payment-network integration. All ML metrics below are historical, not new hardening results. Local hardening did not retrain the frozen model or regenerate evaluation reports. See [verification](verification.md) for current backend/frontend checks and execution evidence; do not memorize changing test counts.

## Two-minute pitch

> I built RiskForge to connect fraud modeling with the engineering required to make a traceable transaction decision. It uses 590,540 public IEEE-CIS transactions and a chronological train, validation, and test split rather than a random split.
>
> The model contract has 28 inputs: 13 behavioral-family features, four categorical fields, and eleven supplied C/D fields. The shared feature builder only uses strictly earlier transactions, excludes simultaneous events, and keeps training and serving feature logic aligned. I compared Logistic Regression, Random Forest, and XGBoost. The historical raw XGBoost TEST Average Precision was 0.449254. Isotonic calibration improved probability quality, but calibrated AP was lower, at 0.438919; I keep ranking and calibration claims separate.
>
> FastAPI loads a frozen artifact, builds features from caller-supplied history, calibrates the score, applies fixed ALLOW/REVIEW/BLOCK thresholds, and returns raw-margin TreeSHAP signals separately from the policy explanation. Before returning success, it commits a PostgreSQL decision record. Optional request idempotency makes scoring retries replay the original response. REVIEW outcomes create cases that can be resolved once under a row lock, with an append-only event and database transition guards.
>
> The limitations are explicit: synthetic policy economics, unknown original threshold rationale, a local static reviewer credential, and no complete score replay from the audit database because prior history is not retained. The strongest result is a testable separation of modeling, policy, and persistence—not a claim about live fraud reduction.

Metric sources: [benchmark CSV](../reports/model_benchmark/model_metrics.csv), [run metadata](../reports/model_benchmark/run_metadata.json), and the [historical calibration record](model-evaluation.md#calibration). Calibration report availability is qualified below.

## Five-minute architecture walkthrough

| Time | Draw or explain | Important boundary |
| --- | --- | --- |
| 0–1 min | IEEE-CIS → point-in-time features → timestamp-grouped TRAIN/VALIDATION/TEST | TRAIN fits preprocessing and estimators; VALIDATION selects; TEST reports. Earlier unlabeled events may supply later history, never future events or labels. |
| 1–2 min | Preprocessor → XGBoost → isotonic calibrator → frozen artifact | Imputation statistics and category vocabulary come from TRAIN; the calibrator is fitted on VALIDATION scores. MLflow tracks offline experiments, not online requests. |
| 2–3 min | Request → validation → shared feature builder → score/calibration → policy + TreeSHAP | Caller supplies history; the API does not query PostgreSQL for features, train, or tune thresholds. The React UI calls this path rather than duplicating it. |
| 3–4 min | PostgreSQL transaction: decision + optional review case + optional idempotency snapshot → response | Persistence is synchronous and must succeed before success is returned. Show where the unique key resolves competing scoring writers. |
| 4–5 min | Review queue → static-key check → row lock → terminal state + event | Human outcome does not overwrite the automated REVIEW decision. Explain failure responses, incomplete replay evidence, and what remains outside the prototype. |

Startup loads and validates the local artifact once (`xgb-001`, `behavioral-v1`, `policy-001`), including agreement with the locked thresholds. A missing/incompatible artifact leaves health reporting available but scoring unavailable; there is no fallback model. SQLAlchemy opens connections lazily, uses a session per repository operation, and disposes the owned pool at shutdown. Alembic owns schema migrations. See [system design](system-design.md), [startup code](../app/dependencies.py), and [repository](../db/repository.py).

## ML chronology and the exact feature contract

1. Sort by elapsed `TransactionDT`; build history using `t − window < event_time < t`. Score every row in a timestamp group before inserting that group into history. Card history uses `card1`; address history uses `(addr1, addr2)`.
2. Split on distinct timestamps: TRAIN **414,446**, VALIDATION **88,013**, TEST **88,081** rows. The nominal split is 70/15/15; ties stay together. Counts and feature membership are in [benchmark metadata](../reports/model_benchmark/run_metadata.json).
3. Fit preprocessing and each estimator on TRAIN only. Numeric gaps use TRAIN medians plus fitted missing indicators; categorical gaps use `__MISSING__` plus one-hot encoding, with unseen categories ignored. XGBoost's positive-class weight is TRAIN negatives divided by TRAIN positives.
4. VALIDATION informed estimator comparison, removal of `D5`, calibration selection, and policy research. It is a reused selection partition, not an independent generalization estimate. Forward chaining does not erase that selection bias; no nested selection evaluation is implemented.
5. Compare sigmoid and isotonic calibration across five chronological VALIDATION folds: first fold warms up, later folds use calibrators fitted on earlier folds. Select isotonic on OOF Brier, then refit it on all VALIDATION scores. The underlying XGBoost estimator remains TRAIN-fitted.
6. Report the frozen candidates/calibration on TEST. Do not silently change the model or thresholds to improve the quoted holdout result. See [evaluation methodology](model-evaluation.md).

The authoritative order and membership are in [ml/data/modeling.py](../ml/data/modeling.py), with derivation details in [feature_contract.csv](../ml/data/feature_contract.csv):

| Family | Count | Contract fields |
| --- | ---: | --- |
| Behavioral | 13 | `TransactionAmt`, `amount_log1p`, `transaction_hour_bucket`, `transaction_day_mod7`, `card_txn_count_1h`, `card_txn_count_24h`, `card_has_history_1h`, `card_has_history_24h`, `card_amount_sum_24h`, `address_txn_count_1h`, `address_txn_count_24h`, `address_has_history_1h`, `address_amount_sum_24h` |
| Categorical | 4 | `ProductCD`, `card6`, `P_emaildomain`, `R_emaildomain` |
| C/D numeric | 11 | `C1`, `C2`, `C11`, `C13`, `D2`, `D3`, `D4`, `D8`, `D10`, `D11`, `D15` |

**Do not say “engineered 28 behavioral features.”** This is a 28-input contract containing a 13-feature behavioral family, which itself includes the original amount. Entity keys and `TransactionDT` support construction/splitting but are not model inputs. One-hot columns and missing indicators expand the transformed representation beyond 28. Elapsed hour/day buckets are not verified calendar-hour/weekday semantics. Missing entity keys produce zero history for that entity family, not evidence that the customer is safe.

### Ranking is not calibration

| Historical locked TEST result | Raw XGBoost | Isotonic-calibrated |
| --- | ---: | ---: |
| Average Precision | 0.449254 | 0.438919 |
| ROC-AUC | 0.865318 | 0.864850 |
| Brier score | 0.101228 | 0.024204 |
| Log loss | 0.344462 | 0.102851 |

- **AP:** the report column named `pr_auc` uses scikit-learn `average_precision_score`: precision weighted by recall increments, not arbitrary trapezoidal PR area. “0.449 TEST AP” refers to the **raw** score. Accuracy is not the primary measure for rare fraud.
- **Calibration:** Brier measures squared probability error; log loss penalizes confidently wrong predictions. Isotonic won the stated OOF Brier criterion, not OOF log loss. Its non-decreasing step function can create ties, explaining why improved probability quality need not improve AP or ROC-AUC.
- **Comparison:** XGBoost led raw TEST ranking metrics; Random Forest had better raw Brier/log loss. Do not claim XGBoost won every metric.
- **Evidence:** raw results are in [model_metrics.csv](../reports/model_benchmark/model_metrics.csv). Calibrated values above are the committed historical table in [model evaluation](model-evaluation.md#calibration); `reports/calibration/evaluation_metrics.json` is generated on demand and is **not committed**. Do not imply it exists in a clean checkout or invent percentage improvements.

### Temporal experiments and uncertainty

The [expanding-window experiment](temporal_evaluation.md) is separate and descriptive. Each fold trains strictly before its own test window, but later training windows overlap the locked benchmark partitions: fold 3 includes **58,741** locked VALIDATION rows; fold 4 includes **29,419** locked TEST rows. Its mean is neither a replacement for 0.449254 nor independent corroboration on a fresh dataset. Sources: [fold report](../reports/temporal_robustness/walk_forward_folds.csv) and [configuration](../reports/temporal_robustness/run_metadata.json).

The locked benchmark's [paired bootstrap](../reports/model_benchmark/test_bootstrap_intervals.csv) resamples transactions, not time blocks. The separate [time-block bootstrap](../reports/temporal_robustness/test_time_block_bootstrap.csv) addresses local temporal dependence only conditionally on its chosen blocks and fixed holdout. Neither establishes future performance or captures all selection/retraining uncertainty. Temporal report floats were not freshly re-derived during hardening.

## Frozen policy and honest provenance

| Action | Calibrated probability |
| --- | --- |
| ALLOW | `p < 0.075` |
| REVIEW | `0.075 <= p < 0.110` |
| BLOCK | `p >= 0.110` |

These are frozen in [policy/locked.py](../policy/locked.py), **not runtime-configurable API thresholds**. Artifact mismatch is rejected. The original selection rationale for the frozen values, especially `0.075`, is unknown. Do not justify them retrospectively as optimal.

A separate historically documented cost-search candidate is **0.050 / 0.110**, selected using VALIDATION under synthetic missed-fraud cost 100, false-decline cost 10, review cost 5, and assumed review fraud capture 0.70. It was **not adopted**, and the reason is also unknown. The search is print-only; the [provenance JSON](../reports/policy_provenance.json) is a retrospective evidence/gap record, not an original run artifact. The policy is amount-blind and the assumed economics are not measured payment losses.

## Explaining a decision without making causal claims

[TreeSHAP](../explain/model_explanation.py) explains the fitted XGBoost model in **raw-margin/log-odds space**, using tree-path-dependent attribution. Base value plus **all** contributions reconstructs the raw margin within numerical tolerance; sigmoid maps that margin to the raw probability, then isotonic maps to the served probability.

[Attribution mapping](../explain/feature_attribution.py) sums one-hot and missing-indicator contributions back to the 28 contract inputs. A displayed top-k subset is not the complete additive decomposition. A positive contribution is an upward raw-model signal, not a calibrated probability-point increase, proof of fraud, or causal effect of changing that field. Isotonic can flatten a score interval. The separate policy explanation identifies the calibrated threshold band—not why the model learned an association or why the threshold was selected.

## API and database lifecycle: retries versus review resolution

| Endpoint | What to say |
| --- | --- |
| `GET /health` | Artifact identity and database readiness; inspect the body, not just HTTP 200. |
| `POST /v1/risk/score` | Transaction, prior `history`, optional `top_k`; optional `Idempotency-Key`. |
| `GET /v1/audit/decisions` | Bounded, newest-first history, with optional action filter. |
| `GET /v1/reviews` | Static-key-protected queue; defaults to PENDING, supports terminal-status filters. |
| `POST /v1/reviews/{case_id}/resolve` | Static-key-protected APPROVED/REJECTED outcome, optional comment; one transition only. |

Requests reject same-time/future history and more than 2,000 history records rather than silently truncating them. Only trailing 24-hour history affects behavioral features. The caller must supply complete, authentic, correctly associated history; the service cannot establish that. See [API contract](api.md).

### Scoring idempotency: replay the successful response

1. Validate the body and header. Keys accept 8–128 characters from letters, digits, `.`, `_`, `:`, `-`.
2. Canonicalize the validated body with sorted JSON object keys and compact separators; compute **SHA-256**. The fingerprint includes transaction, history, and `top_k`. It is not an authentication signature; history-array order is not normalized by JSON key sorting.
3. Existing key + same hash returns the original stored response without new inference or another decision. Same key + different hash returns **409**. The route still requires an available loaded service and database.
4. On a miss, score and atomically commit `risk_decisions`, a `risk_review_cases` row if REVIEW, and `risk_idempotency_records` containing the hash and response snapshot.
5. A pre-check alone cannot prevent a race. The unique key constraint elects one writer; the loser's failed transaction rolls back **all** its decision/case writes. A fresh session rereads the winner: matching hash replays, mismatching hash conflicts; failure to recover a winner is an audit failure, not success.

Keys are **global to this single-operator prototype and retained indefinitely**. There is no tenant scope or expiry policy. Without a key, repeated requests may create separate decisions; `transaction_id` is indexed, not unique. Racing requests can compute twice even though only one decision commits. This is database-effect deduplication, not an exactly-once execution guarantee. Code: [route](../app/main.py), [repository](../db/repository.py).

### Human review: one state transition, not successful-response replay

- Only automated REVIEW creates a case. Resolution acquires `SELECT ... FOR UPDATE`, checks PENDING, updates the case and appends a review event in one transaction. The original automated decision remains REVIEW.
- A PostgreSQL one-way trigger independently rejects invalid/repeated transitions; hardened guards protect identity and resolution metadata and prevent case deletion. Decision, review-event, and idempotency tables reject UPDATE/DELETE. These are protections against ordinary application writes, not tamper-proofing against an administrator who can alter triggers.
- Concurrent resolvers yield one committed outcome/event. A repeated resolution—even the same outcome from the same reviewer—returns **409**, not its original success body. There is no resolution idempotency-key contract.
- After an ambiguous timeout, query `GET /v1/reviews?status=APPROVED` or `status=REJECTED`, paginate to the case, and inspect its recorded outcome. Absence from the default pending queue alone does not identify the outcome. With a shared static identity, this still cannot prove which human or retry won.
- Reviewer authentication is a local static `X-Reviewer-API-Key`, compared using `hmac.compare_digest`, with a configured `RISKFORGE_REVIEWER_ID`. It is **not real user identity or RBAC**. Scoring and decision-history reads are unauthenticated local-demo routes.

The database saves transaction details, probability, action, explanations, and model/feature/calibration/policy identifiers. **Prior-history rows are not saved**, so neither an audit row nor the request hash allows full feature/score reconstruction. Stored response replay is different from recomputing a score.

## Failures to rehearse

| Situation | Actual behavior / answer |
| --- | --- |
| Invalid body, non-prior history, oversized history | 422; fix the request. Body validation keeps FastAPI's `detail` form. |
| Artifact missing, invalid, or policy-incompatible | Scoring 503; no fallback training or fabricated score. |
| Audit write/read or review persistence failure | 503 `audit_unavailable`; no successful unaudited scoring response. A commit followed by a lost response remains ambiguous—retry scoring with the original key. |
| Unexpected inference error | 500 `inference_failed`, with no probability/decision. Internal exception details stay server-side. |
| Reused key with different body / already resolved case | 409 conflict; do not blindly change a key to bypass the conflict. |
| Missing/wrong reviewer key / credentials unconfigured | 401 / 503 respectively; unknown review case is 404. |
| Healthy artifact but unavailable DB | `/health` can return 200 with `database="unavailable"`; this is not scoring readiness. |

Most application failures use a sanitized `error.code` / `error.message` envelope; 422 and health have distinct shapes. Do not promise that a network timeout means “nothing committed.”

## Likely Visa / PayPal follow-ups

**Why AP instead of accuracy or just ROC-AUC?** Fraud is rare; AP focuses on positive-class precision/recall across score thresholds. ROC-AUC complements it but does not tell us review precision, false-decline volume, or operational cost at a chosen threshold. AP also depends on prevalence; it is not directly transferable across traffic populations.

**Why XGBoost, and why not simply use its probability?** It had the strongest historical ranking among these fixed candidates; weighted classification scores are not automatically reliable probabilities. Validation-selected isotonic addresses probability quality, while acknowledging lower calibrated AP and calibration drift risk.

**Can you prove there is no training-serving skew or leakage?** Shared code, strict timestamp boundaries, and tests support the design; a fresh parity check passed three real VALIDATION cases, one per action. That is not exhaustive proof over all inputs/histories, nor evidence that supplied C/D fields would be available in a real payment feed. Verify feature availability before transfer.

**How would you choose thresholds for our business?** Obtain measured loss/false-decline costs, amount exposure, review capacity, and review effectiveness; choose on time-separated selection data, then evaluate on untouched future data. Those are proposed next steps, not implemented evidence. The current frozen rationale is unknown and the synthetic search candidate was not adopted for an undocumented reason.

**What happens when two retries or reviewers race?** Scoring uses atomic writes plus unique-key conflict rollback and winner reread. Review uses row locking plus a one-way trigger. The former replays success; the latter rejects repeats with 409. Neither mechanism alone provides enterprise identity, tenant isolation, or guaranteed request delivery.

**Why synchronous PostgreSQL rather than an asynchronous audit queue?** It makes “successful score has a committed record” straightforward, at the cost of coupling scoring availability and response time to the DB. An asynchronous design would require durable handoff, deduplication, and explicit failure semantics; it is not implemented. No load or latency claim is supported here.

**How would you handle fraud drift, delayed chargebacks, and review feedback?** Propose monitoring score/input distributions, missingness, action volumes, and eventually time-matured labeled AP/calibration by segment. Human APPROVED/REJECTED is not automatically ground-truth fraud. Delayed outcomes and selectively observed labels require careful evaluation; there is no live monitoring or automatic retraining loop in this project.

**What is missing before using this with payment traffic?** Trusted point-in-time history and feature provenance, retained/replayable inputs under a privacy policy, real authentication/authorization, tenant-scoped keys and retention, security controls, load testing, and operational recovery. The public-data prototype does not establish bank performance, regulatory compliance, financial savings, or complete audit replay.

## Resume claim evidence and safe wording

| Target claim | Evidence to open | Boundary / correction |
| --- | --- | --- |
| “590K transactions, 28 features” | [README](../README.md#dataset-and-point-in-time-features), [benchmark metadata](../reports/model_benchmark/run_metadata.json), [contract](../ml/data/modeling.py) | 590,540 public historical rows; 13 behavioral + 4 categorical + 11 C/D. Not 28 newly engineered behavioral features. |
| “0.449 TEST AP” | [benchmark CSV](../reports/model_benchmark/model_metrics.csv), [metric implementation](../ml/models/evaluation.py) | Rounded **raw** AP 0.449254 via `average_precision_score`, not arbitrary PR area. Calibrated AP is 0.438919. |
| “Calibrated probabilities” | [historical calibration table](model-evaluation.md#calibration); on-demand path `reports/calibration/evaluation_metrics.json` | Isotonic selected on OOF Brier, not every metric. The JSON is not committed; no fresh calibration run or unstated percentage claim. |
| “Policy-driven FastAPI scoring” | [locked policy](../policy/locked.py), [service guard](../app/service.py) | Frozen 0.075/0.110, not runtime configurable. Original rationale and non-adoption reason for 0.050/0.110 are unknown. |
| “Auditable, idempotent scoring” | [repository](../db/repository.py), [API persistence tests](../tests/test_app_audit_api.py), [PostgreSQL tests](../tests/test_db_repository.py) | Atomic decision/case/snapshot writes, SHA-256 matching, rollback/winner reread supported. Global indefinite keys; response replay is not full score reconstruction. |
| “Concurrency-safe human review” | `test_concurrent_reviewers_produce_one_resolution_and_one_event` in [DB tests](../tests/test_db_repository.py), [transition migration](../alembic/versions/d8f4b2c6a901_harden_review_case_transition_guard.py) | One committed resolution/event, repeated calls 409. Static reviewer key, not multi-user identity/RBAC. |
| “Temporal robustness evaluation” | [fold reports](../reports/temporal_robustness/walk_forward_folds.csv), [temporal methodology](temporal_evaluation.md) | Separate descriptive experiment; expanding training windows overlap locked holdout rows. Not an independent higher TEST score. |

### Four precise resume bullets

- Built an offline fraud-risk pipeline over 590,540 IEEE-CIS transactions with chronological 70/15/15 evaluation and a fixed 28-input contract: 13 behavioral, four categorical, and eleven C/D features.
- Benchmarked Logistic Regression, Random Forest, and XGBoost; recorded historical raw XGBoost TEST Average Precision of 0.449254 and validation-selected isotonic calibration with historical TEST Brier score of 0.024204.
- Implemented FastAPI transaction scoring with frozen ALLOW/REVIEW/BLOCK policy thresholds, raw-margin TreeSHAP signals, and versioned PostgreSQL decision audit records.
- Added SHA-256 request idempotency with atomic response replay and one-time human review using row locks, append-only events, and PostgreSQL concurrency tests.

**Before the interview:** open [verification](verification.md) for the latest backend and frontend check status. Fresh backend checks and three real VALIDATION parity cases passed during local hardening; parity is targeted, not exhaustive. Keep this fresh engineering evidence separate from historical ML reports and do not claim a fresh frontend rerun unless that record confirms it.
