# Release verification

This document records the checks used before presenting RiskForge as a completed portfolio project. It distinguishes automated repository checks from local checks that require the private IEEE-CIS dataset and generated serving artifact.

## Final release state

- Default branch: `main`
- Final benchmark, robustness reports, architecture/workflow diagrams and technical-defense documentation were merged in [PR #5](https://github.com/Sujeet-Kona/RiskForge/pull/5).
- The final PR head passed [RiskForge CI run 41816085951](https://github.com/Sujeet-Kona/RiskForge/actions/runs/41816085951) before merge. This run completed lint, compilation, PostgreSQL migrations/schema checks and the full test suite successfully.
- The merge commit is `fbc2aceb1c7f22eb4fd68e693db3e0cd58edfc48`. The main-branch workflow runs after the merge; the live CI badge in the README links to the latest main-branch result.
- Local artifact parity was manually run on 2026-11-04 against `xgb-001 / behavioral-v1 / policy-001`, as detailed below.

The dataset and fitted artifact are intentionally not committed. Reproducing the data/model workflows requires obtaining IEEE-CIS locally and rebuilding the artifact using the documented commands.

## Offline/API parity result

Command:

```bash
PYTHONPATH=. python scripts/verify_serving_parity.py
```

Observed result:

- The script loaded the published local serving artifact and rebuilt the chronological split.
- The locked validation statistics reproduced: calibrated mean `0.034199`; ALLOW `81,584`, REVIEW `2,235`, BLOCK `4,194`.
- Three validation transactions were selected from calibrated probability and locked thresholds only: one ALLOW, one REVIEW, and one BLOCK. Labels were not used to select them.
- All 28 feature values matched between the offline and API paths for all three cases.
- Offline and API probabilities, decisions, reason codes, and rendered decision explanations matched.
- The two floating-point amount-sum features were within the configured tolerance: worst absolute delta `1.310e-10`, tolerance `1e-9`.
- The script did not train, recalibrate, tune thresholds, or score the TEST partition.

**Interpretation:** this verifies parity for three representative cases under the supplied-history contract. It is not an exhaustive proof over all possible requests, and it does not validate the authenticity or completeness of caller-supplied history.

## Reproduction checklist

Run from the repository root with the local virtual environment active.

```bash
# Static checks
./.venv/bin/python -m ruff check .
./.venv/bin/python -m compileall -q .

# Database schema and integration tests
./.venv/bin/alembic upgrade head
./.venv/bin/alembic check
RISKFORGE_REQUIRE_POSTGRES=1 ./.venv/bin/python -m pytest -q

# Requires the local IEEE-CIS data and generated serving artifact
PYTHONPATH=. ./.venv/bin/python scripts/verify_serving_parity.py
```

CI runs Ruff, compilation, Alembic checks, and the pytest suite against PostgreSQL 17. CI does not download IEEE-CIS, train the model, build the serving artifact, run offline/API parity, or deploy the service.

## Known boundaries to preserve in the presentation

- IEEE-CIS is an offline public benchmark, not live payment-network data.
- TEST metrics represent one chronological holdout, not future production performance.
- Policy costs and review-effectiveness assumptions are synthetic; the experimental candidate is not the active serving policy.
- The active API policy remains `policy-001` with review threshold `0.075` and block threshold `0.110`.
- The API relies on caller-supplied prior history and cannot verify its completeness, freshness, or authenticity.
- Audit rows do not store all prior-history rows needed to replay a score.
- Append-only behaviour is an application convention, not a database-enforced immutability guarantee; idempotency/deduplication is not implemented.
- No production deployment, authentication, TLS termination, rate limiting, drift monitoring, or human-review operation is claimed.

These are explicit project boundaries, not hidden implementation details. Keep them visible in technical discussions and do not describe RiskForge as a production payment fraud platform.
