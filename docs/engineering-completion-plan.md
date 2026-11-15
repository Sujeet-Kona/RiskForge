# RiskForge engineering completion plan

## Baseline

Audit date: 2026-11-15
Base branch: `main`
Base commit: `37883498801fb998f1d35d6c8ec7d2e20b6cf271`
Working branch: `feat/riskforge-operations-console`

This plan is based on the current repository tree, README, API schemas, serving lifecycle, audit repository, database model/migration, CI workflow, and temporal-evaluation documentation. It is a scoped engineering audit, not a claim that every source file or local runtime has been executed in this environment.

## Confirmed existing capabilities

- Chronological IEEE-CIS modeling pipeline with 590,540 rows and a 28-feature contract.
- Point-in-time behavioral feature construction and strict-prior history validation at the API boundary.
- XGBoost training, forward-chaining calibration research, locked isotonic serving artifact, and a deterministic ALLOW / REVIEW / BLOCK policy.
- TreeSHAP model-signal explanations kept separate from policy-threshold explanations.
- FastAPI health and scoring endpoints; scoring fails closed when the artifact is unavailable.
- PostgreSQL decision audit write occurs before a successful scoring response; model, feature, and policy versions are recorded.
- SQLAlchemy/Alembic schema and an existing pytest/Ruff/PostgreSQL CI workflow.
- Temporal walk-forward and moving-time-block bootstrap tooling and focused tests are present in the repository.

## Confirmed gaps / constraints in the current documented implementation

- No frontend exists in the current repository tree.
- No transaction-list/dashboard API exists; the current API exposes health and score.
- No human-review case queue or reviewer resolution workflow exists.
- No idempotency key or duplicate-request suppression exists.
- No authentication/authorization, TLS termination, or rate limiting is implemented.
- Audit rows do not retain prior-history rows, so the stored record alone cannot exactly replay a score.
- Append-only audit behavior is an application convention, not database-enforced immutability.
- No operational drift monitoring or production deployment is implemented.
- The README says temporal robustness outputs require the IEEE-CIS source data and does not claim additional robustness metrics until those runs complete.

## Evidence issue requiring resolution

The README reports calibrated TEST Brier = 0.024204 versus raw XGBoost TEST Brier = 0.101228, approximately a 76.1% reduction. It explicitly states that the raw TEST ECE baseline is not included in the committed benchmark CSV and therefore does not claim a percentage ECE reduction. The resume's 98% ECE-reduction claim must be supported by a reproducible saved calculation/report or removed from future resume wording. Do not infer or fabricate the baseline.

## Design principles for the completion work

1. Preserve the existing model contract, serving artifact format, scoring semantics, decision thresholds, and current API response contract unless a migration is justified and tested.
2. Add application functionality around the scorer; do not duplicate feature construction or model inference in the browser.
3. Keep automated model decisions immutable. Human review outcomes are separate records and never overwrite the original ALLOW / REVIEW / BLOCK result.
4. Use explicit idempotency keys for retry-safe scoring; do not make transaction_id unique because repeat scoring may be legitimate.
5. Persist model, feature, and policy versions with every automated decision and reviewer action.
6. Make authentication configurable for local development, fail closed when enabled, and document that demo authentication is not production identity infrastructure.
7. Never display synthetic or hard-coded operational metrics as live data. Label demo fixtures clearly.
8. Work on this feature branch; do not push changes directly to main. Merge only after CI and local integration checks pass.

## Planned implementation sequence

### Milestone 1 — Backend contracts and persistence
- Add paginated decision listing and decision detail endpoints.
- Add a separate review-case lifecycle for REVIEW decisions and append-only review actions.
- Add reviewer authentication/authorization suitable for a local portfolio demo, with explicit production limitations.
- Add idempotency-key persistence and conflict semantics for score requests.
- Add tests for migration, persistence, duplicate retries, permissions, and review state transitions.

### Milestone 2 — Professional frontend
- Add a React + TypeScript + Vite application.
- Use a restrained premium fintech design system: deep ink/navy surfaces, warm off-white content canvas, restrained teal/blue accent, and semantic green/amber/red risk states.
- Build Overview, Transactions, Transaction Detail, Review Queue, Review Case, and System/Model pages.
- Consume only the FastAPI contracts; provide loading, empty, validation, offline, and error states.
- Clearly distinguish actual API data from seeded demo data.

### Milestone 3 — End-to-end quality
- Add frontend type-check, lint, and production build to CI.
- Add API/database integration and browser-level smoke tests where practical.
- Test ALLOW, REVIEW, BLOCK, human resolution, duplicate retries, missing artifact, database outage, invalid history, and unauthorized review actions.
- Re-run existing ML/backend tests without modifying locked benchmark behavior.

### Milestone 4 — Evidence and interview readiness
- Reconcile every resume metric against committed reports and reproducible commands.
- Run the temporal robustness experiment when the private/local IEEE-CIS files are available; save metadata and outputs without committing the dataset.
- Update README, system design, API examples, local setup, screenshots, and a concise demo script.
- Prepare an interview walkthrough covering modeling choices, calibration, temporal evaluation, decision policy, SHAP limitations, auditability, idempotency, review lifecycle, and operational boundaries.

## Verification gates

A milestone is not complete until:
- Changes are committed on this feature branch.
- Relevant automated tests pass.
- GitHub Actions passes for the branch/PR.
- Existing API and ML behavior is regression-tested.
- Documentation describes only implemented and verified behavior.

## Local-only verification boundary

This remote audit cannot access the developer's Mac, local virtual environment, local Docker volume, Kaggle dataset, serving artifact, or secrets. Any test or experiment requiring those resources must be run locally and its output reviewed before the related capability is marked verified.
