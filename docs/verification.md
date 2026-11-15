# Local engineering verification — 2026-11-15

## Executive status

**VERIFIED:** the current local backend, database, serving-parity and frontend
checks listed below, and dashboard acceptance of one APPROVE and one REJECT
manual-review case (see [manual-review dashboard acceptance](#manual-review-dashboard-acceptance-2026-11-16)).
PRs #10–#13 were merged into `main` in dependency order on 2026-11-16, with this
completion branch (PR #14) as the final step (see [GitHub integration](#github-integration)).
No independent human code review approval is recorded on those PRs. This is not
a declaration that the prototype is production-ready.

Work was isolated on `hardening/riskforge-completion`, based on the existing PR
stack, with PR #11's required fix merged into the stack tip. Final checks covered
current tracked working-tree contents after base commit `3b38786`, not just that
commit. Use Git history for the subsequent local completion commit.

## Defects corrected and affected areas

| Original gap | Local correction | Principal files |
| --- | --- | --- |
| Missing threshold rationale and provenance artifact | Record known facts and unknowns without inventing a rationale; leave policy unchanged | `policy/locked.py`, `reports/policy_provenance.json`, `docs/model-evaluation.md` |
| Descriptive feature contract disagreed with selected inputs | Align documentation with authoritative `MODEL_FEATURES`; add membership/status tests | `ml/data/feature_contract.csv`, `tests/test_modeling.py` |
| Reproduction and validation claims were too broad | Separate selection reuse, historical reports, optional research and unavailable inputs | `README.md`, `docs/reproducibility.md`, `docs/temporal_evaluation.md` |
| Concurrent review resolution lacked a direct regression | Six synchronized workers require one terminal outcome and one event | `tests/test_db_repository.py` |
| Application errors used inconsistent shapes | Static-message envelope with explicit health/422 exceptions | `app/main.py`, `tests/test_app_audit_api.py`, `docs/api.md` |
| Parity verification could use a persistent audit path | Keep parity decisions in an in-memory audit store | `scripts/verify_serving_parity.py` |
| UI hardcoded feature count and weak failure states | Read artifact count; preserve validation reasons; distinguish loading, failure, authenticated empty queue and success | `frontend/src/App.tsx`, `frontend/src/api.ts`, `frontend/src/types.ts` |
| Credential changes retained prior review details | Clear cases/comments/notices on key change; lock credentials during in-flight operations | `frontend/src/App.tsx` |
| Small/low-contrast text, weak keyboard/status behavior and mobile overflow | Typography/contrast/focus improvements, persistent status regions, skip link, wrapping and bounded grid panels | `frontend/src/index.css`, `frontend/src/App.tsx` |
| Different model features can share a reason code | Key rendered signals by direction and feature instead of reason-code category | `frontend/src/App.tsx`, `frontend/src/App.test.tsx` |
| No frontend regression suite | Vitest + Testing Library + jsdom, API fetch tests and focused workflow tests | `frontend/package.json`, `frontend/package-lock.json`, `frontend/vitest.config.ts`, `frontend/src/test/setup.ts`, `frontend/src/*.test.*` |
| Interview/resume/API guidance incomplete | Concrete contract examples, evidence-qualified claims and interview answers | `docs/api.md`, `docs/interview-guide.md`, `docs/system-design.md` |

The original checkout's tracked changes and untracked lockfile, reports and
backup script were preserved. The original Vite `^6.4.6` change was retained.
Dataset files were not staged or modified by feature-generation scripts; the
completion checkout's `data` is a symlink and was treated as read-only. The local
serving artifact remains byte-identical to the original copy:

```text
SHA-256 b6dd06fd9b5770e74ab1f02415a93c74ba2cbb3b7c4ab97aa3f66123959f77ca
```

## Executed backend and database checks

Commands ran from the completion repository with its `PYTHONPATH`, the existing
Python 3.12 virtual environment, and `RISKFORGE_REQUIRE_POSTGRES=1`.
`DATABASE_URL` explicitly targeted the disposable local `riskforge_test` database
on `127.0.0.1:5432`. No migration or test reset targeted the user's application
database. See [test database precautions](reproducibility.md#local-api-and-database-checks)
before repeating these commands: fixtures rebuild and truncate the test database.

| Command | Exit | Inspected result |
| --- | ---: | --- |
| `python -m ruff check .` | 0 | All checks passed |
| `python -m compileall -q app db ml policy explain tracking scripts tests alembic` | 0 | Compilation completed |
| `python -m pytest -q` | 0 | **1,256 passed**, 0 failed, 0 skipped; 1 warning; 12.35 s |
| `python -m pytest -q tests/test_db_migration.py` | 0 | **16 passed**, 0 failed/skipped; 0.89 s |
| `python -m pytest -q tests/test_db_repository.py -k concurrent` | 0 | **3 passed**, 31 deselected; 0.57 s |
| `python -m pytest -q tests/test_decision_policy.py` | 0 | **40 passed**, 0 failed/skipped; 0.35 s |
| `python -m alembic check` | 0 | No new upgrade operations detected |
| `python scripts/verify_serving_parity.py` | 0 | Three real VALIDATION cases agree; 1.41 s |
| `python -m pip check` | 0 | No broken requirements found |

The focused runs repeat subsets of the full suite; their counts must not be
added to 1,256 as if they were distinct tests. PostgreSQL failures were not
converted to skips. The warning is the existing Starlette deprecation of
`httpx` with `TestClient` in favor of `httpx2`; it was not suppressed or hidden by
a dependency downgrade. `pip check` tests dependency compatibility, not security
advisories.

### Artifact-free source verification

An initial temporary source copy omitted `.git`. It produced **1,255 passed,
1 failed**, exit 1: `test_the_commit_sha_is_recorded_as_a_tag_and_in_the_metadata`
correctly received `unavailable` instead of a Git SHA. This was a verification
harness mistake, not a reason to weaken the test.

The corrective local Git clone, with current tracked source contents overlaid,
no IEEE-CIS dataset and no serving artifact, completed `python -m pytest -q` with
**1,256 passed, 0 failed/skipped, 1 warning**, exit **0**, in **11.80 s**.
`clean-git.log` and `clean-git-result.json` retain that result separately from the
failed source-copy attempt. This uses the existing Python environment; it is not
a fresh operating-system or Python dependency installation.

## Executed frontend checks

The test stack uses Vitest 4.1.12, jsdom 29.1.1 and React Testing Library. It adds
no browser-test runner, API mocking server or coverage package. The package
lockfile preserves Vite 6.4.6. See [runtime requirements](../frontend/README.md).

Commands ran from `frontend/` and again in a temporary source copy with its own
fresh `node_modules`, without replacing the running development server's modules.

| Command | Exit | Inspected result |
| --- | ---: | --- |
| `npm test` | 0 | **60 passed**: 42 component + 18 API tests, 2 files, no skips; 1.98 s |
| `npm run typecheck` | 0 | TypeScript completed |
| `npm run build` | 0 | TypeScript + Vite production build completed |
| `npm audit --audit-level=low` | 0 | 0 reported vulnerabilities |
| `npm ci` in isolated frontend copy | 0 | 155 packages installed; 156 audited; 0 reported vulnerabilities |
| `npm test` in isolated copy | 0 | **60 passed**, 2 files, no skips; 1.83 s |
| `npm run typecheck` in isolated copy | 0 | TypeScript completed |
| `npm run build` in isolated copy | 0 | Same emitted bundle names/sizes as working frontend |

Production output: `index-BoytjA8x.css` 26.90 kB (gzip 6.69 kB),
`index-ChvX6TbO.js` 177.14 kB (gzip 54.48 kB), HTML 0.57 kB.
These are build sizes, not measured service latency or throughput.

The duplicate-reason-code regression was first run against the broken keying:
exit 1, one failed test and 41 filtered-out tests (reported as skipped by Vitest).
After keying by feature/direction, the unfiltered 60-test suite passed. No test
was disabled to obtain that result.

## Browser and end-to-end workflow evidence

The actual Vite application ran on loopback port 5173, proxied to FastAPI on
8001 using only `riskforge_test`. Browser automation and read-only SQL inspection
verified the following; this is a session walkthrough, not a committed unattended
browser-E2E runner or user acceptance.

- Real API readiness and artifact feature count; actual scoring, calibrated
  probability, thresholds, model identities, explanations and persisted history.
- A real VALIDATION REVIEW example was submitted with its original 29 strictly
  prior history rows. The model returned probability `0.09090909090909091` and
  REVIEW. Retrying the identical idempotent payload returned an identical 200
  response and left one decision/case.
- The browser loaded that case, recorded an APPROVED local-verification outcome,
  displayed success and **“No pending review cases.”** SQL confirmed original
  decision REVIEW, case APPROVED, configured reviewer identity and exactly one
  review event. The note explicitly stated this was not an adjudicated fraud
  label. Test-suite resets can subsequently remove these disposable records.
- Earlier walkthroughs verified wrong reviewer credentials and same-timestamp
  history rejection; component/API regressions cover their presentation, loading,
  retries, conflict, empty signals and failed queue refresh after a successful
  resolution. ALLOW/REVIEW/BLOCK rendering uses explicitly synthetic unit fixtures.
- Populated result/history and review panels were measured in real same-origin
  browser frames at **320, 390, 768 and 1440 CSS pixels**. Document width matched
  viewport width; inspected panels and controls did not overflow. Long transaction
  IDs wrap; history tables scroll within their container. This is not a claim
  about every browser or viewport.
- Actual Tab / Shift+Tab revealed the skip link with a 2 px focus ring; Enter
  transferred focus to the main landmark. Persistent status elements and alerts
  were inspected and tested, but no screen-reader user acceptance was performed.

Measured selected text contrast pairs after correction: primary button **5.169:1**,
ALLOW label **5.011:1**, REVIEW label **4.857:1**, BLOCK label **5.914:1**, muted
text on white **5.332:1**, muted text on panel **5.009:1**, risk-reducing signal
**5.291:1**, and placeholder text **4.566:1**. These checks use the rendered CSS
colors and relative luminance; they are not a complete WCAG certification.

Browser console inspection caught duplicate React keys in real model signals;
that defect was reproduced by the failing test above and corrected. A final
post-reload score returned ALLOW at displayed 1.37%, rendered both amount
features sharing the same reason category, and produced **no console warnings
or errors** and no UI alerts.

### Manual-review dashboard acceptance (2026-11-16)

Both cases used the actual Vite dashboard on `127.0.0.1:5173`, proxied to FastAPI
on `127.0.0.1:8001`, with only the disposable `riskforge_test` database. Each case
came from a real VALIDATION transaction scored through `POST /v1/risk/score` with
its strictly prior 24-hour history; case #2's identical idempotent retry replayed
the stored response. The fraud label was not used to select or score either case.
A temporary reviewer key existed only in the API process environment. Afterwards
the API was restarted from the ignored local `.env` and the temporary key returned
HTTP 401.

| Case | Transaction | History rows | Model decision | Dashboard action | Recorded review |
| --- | --- | ---: | --- | --- | --- |
| #2 | 3401505 | 29 | REVIEW, `0.090909` | **Clear transaction** (APPROVE), via browser automation | APPROVED by `reviewer-local`; note “Local acceptance check only; not an adjudicated fraud label.” |
| #3 | 3402682 | 321 | REVIEW, `0.090909` | **Confirm suspected fraud** (REJECT), clicked by the user | REJECTED by `reviewer-local`; **no reviewer note recorded** |

- For each case the dashboard reported the recorded outcome, stated that the
  original model decision remains REVIEW, and showed **“No pending review
  cases.”** Each resolve request returned HTTP 200.
- Read-only SQL confirmed both original decisions remain REVIEW, each transaction
  has exactly one decision and one idempotency record, and each case has exactly
  one review event.
- A second resolution of case #2 through the dashboard proxy returned HTTP 409
  `conflict`; the dashboard offers no action on resolved cases. A second
  resolution of case #3 was not attempted.
- Case #3 was the REVIEW transaction closest to the centre of the REVIEW band
  after excluding previously used IDs. The rebuilt VALIDATION frame matched the
  locked band counts.
- Not verified: the `.env` reviewer key in the dashboard. Reviewer access is a
  single local demonstration key, not production reviewer authentication. These
  disposable records can be removed by test-suite resets. Review outcomes recorded
  here are acceptance actions, not adjudicated fraud labels.

## ML integrity and evidence boundaries

**VERIFIED locally:** shared feature contract, strict-prior boundaries, artifact
loading and frozen policy boundaries, plus the full parity script. It reproduced
VALIDATION band counts ALLOW 81,584 / REVIEW 2,235 / BLOCK 4,194, then compared one
real example per band. All 28 features agreed under the existing rules;
probabilities, decisions, reason codes and policy explanations agreed. The largest
amount-sum difference was `1.310e-10`, below absolute tolerance `1e-9` (relative
`1e-12`); counts/flags required exact agreement. Parity audits remained in memory.

**Historical, not freshly retrained:** benchmark AP/ROC-AUC, calibration results,
bootstrap intervals and temporal experiment floats. No model fitting, feature
regeneration, threshold selection or report overwrite was used to improve
hardening results. Frozen thresholds remain **0.075 / 0.110**. The separate
**0.050 / 0.110** cost-search candidate remains experimental; original rationale
and non-adoption reason are unavailable. The provenance JSON does not fill that
evidence gap. See [model evaluation](model-evaluation.md) for validation reuse and
the small ECE discrepancy, and [temporal evaluation](temporal_evaluation.md) for
non-comparable windows and missing fresh reproduction evidence.

## Security/configuration findings

**VERIFIED by inspection/tests:** request validation, parameterized ORM/database
operations, static-message application errors, constant-time local reviewer-key
comparison, append-only/one-way database protections, failure to return unaudited
scores, in-memory frontend credentials and no credential URL parameter.
The user-owned `.env`, datasets and model binaries were not included in changes.

**NOT IMPLEMENTED:** enterprise identity/RBAC, tenant-scoped key retention,
trusted or replayable history storage, production TLS/rate limiting, production
monitoring, continuous retraining, payment-network integration and load/SLA
validation. Database triggers do not protect against an administrator who can
change them. Human review is not fraud-ground-truth collection. No financial
savings, live transaction volume, production compliance or bank-performance claim
is supported. Python dependency vulnerability scanning was not run; `npm audit`
does not establish whole-system security.

## GitHub integration

Pre-integration snapshot inspected on 2026-11-15:

| PR | Base → head | Head SHA | State / mergeability / checks |
| --- | --- | --- | --- |
| #10 | `main` → `feat/riskforge-application-foundation` | `858f0eea1b5c81e8f5849275e76662a7eaed3ab2` | OPEN / MERGEABLE / two successful checks |
| #11 | foundation → `feat/riskforge-blue-white-polish` | `42ecf4e6e0b79e7f1941b588410e7f938be1e844` | OPEN / MERGEABLE / no reported checks |
| #12 | polish → `test/idempotency-concurrency` | `9a6883e609eba443fe18baa3ea6a3e8aa898d1c0` | OPEN / MERGEABLE / no reported checks |
| #13 | concurrency → `hardening/review-case-transition-guard` | `0fc037a01b381abbd04733e4ad184c0751551036` | OPEN / MERGEABLE / two successful checks |

The completion branch contains both #11's required fix and #13's tip, verified
with `git merge-base --is-ancestor`. No remote completion branch was present.
The original checkout remains on its original local branch/commit with its user
changes intact. A previously authorized push attempt was rejected with HTTP 403
for account `sandy-455`; no fork/remote/authentication setting was changed. The
completion branch was later pushed to `origin/hardening/riskforge-completion` and
opened as draft PR #14 against #13's branch (retargeted to `main` after #13 merged).

Existing remote `main` at `37883498801fb998f1d35d6c8ec7d2e20b6cf271` has a successful
historical CI run from 2026-11-07. That is **not CI for this local work**. CI now
also installs the frontend with `npm ci` and runs Vitest before the type-check and
production build. Run `43753200454` on PR #14 head `a33d55e` passed with 1,256
pytest tests and 60 Vitest tests.

With the repository owner's authorization, the stack was merged into `main` in
dependency order on 2026-11-16 using merge commits, without rewriting any branch.
`main` has no branch protection or required-review rule; no independent human
review approval is recorded. Before each merge, the PR was retargeted to `main`
and its head, mergeability and checks were rechecked. #11, #12 and #13 were
closed and reopened once each solely to trigger CI against `main`, because a
base change alone does not start a workflow run.

| PR | Head | CI run against `main` | Merge commit |
| --- | --- | --- | --- |
| #10 | `858f0ee` | `43719866756`: 1,240 pytest, build passed | `bff1064` |
| #11 | `42ecf4e` | `43758507596`: 1,240 pytest, build passed | `1867b8a` |
| #12 | `9a6883e` | `43758787381`: 1,242 pytest, build passed | `3f9f98a` |
| #13 | `0fc037a` | `43759050635`: 1,246 pytest, migration `d8f4b2c6a901`, build passed | `13843b0` |

Push CI on `main` after each of these four merges also passed (after #13:
run `43759265509` on `13843b0`, 1,246 pytest). PR #14,
which carries this completion branch, is the final step; its merge commit and
the resulting `main` CI run are recorded on GitHub, not in this file, because
this file is committed before that merge.

## Evidence locations and local acceptance

Per-command logs and JSON exit records remain local under the system temporary
directory in `riskforge-backend-release-roxu4w17/` and
`riskforge-frontend-release-j4192vho/`. The earlier source-copy failure is retained,
not overwritten. Browser/SQL walkthrough evidence is in the session tool results.
These temporary logs are not committed release artifacts or guaranteed permanent
storage. The tables above are the inspected verification record.

[Interview guide and resume evidence](interview-guide.md) contains the two-minute
pitch, five-minute architecture, failure/trade-off questions, claim corrections
and four defensible resume bullets. [API examples](api.md) and [setup](reproducibility.md)
cover local usage and database safety.

- [x] Backend, migration, concurrency, threshold and parity commands inspected.
- [x] Frontend tests, typecheck, production build and clean lockfile installation inspected.
- [x] Browser scoring/review workflow and disposable DB persistence inspected.
- [x] Historical metrics distinguished from fresh engineering checks.
- [x] Manual-review dashboard acceptance: case #2 APPROVE and case #3 REJECT; both original decisions remain REVIEW.
- [x] Completion branch pushed as PR #14; CI includes frontend Vitest.
- [x] PRs #10–#13 merged into `main` in dependency order with passing CI against `main`.
- [ ] **NOT PERFORMED:** independent human code review of the PR stack.
- [ ] **NOT IMPLEMENTED:** committed unattended browser-E2E suite.

The API/PostgreSQL integration tests, component tests and browser walkthrough are
different evidence layers. Do not describe the mocked component suite as a full
browser-to-database E2E suite, or treat this checklist as user sign-off.
