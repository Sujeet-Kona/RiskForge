# RiskForge dashboard

React + TypeScript frontend for the existing RiskForge FastAPI scoring service.

## Local development

Use **Node.js 22.13+ on the 22.x line, or a compatible supported Node.js 24+ runtime**.
jsdom 29 requires `^20.19.0 || ^22.13.0 || >=24.0.0`, but jest-dom 7 requires
Node.js `>=22`; Node.js 20 is therefore insufficient for this project's tests.

1. Follow [root local setup](../README.md#run-locally): export the root `.env`,
   wait for a disposable local PostgreSQL server, verify the target database
   before migrations, and complete the minimal serving-artifact build.
2. Start the API from the repository root with that environment exported:

   ```bash
   set -a
   source .env
   set +a
   uvicorn app.main:app --reload
   ```

3. In a second terminal:

   ```bash
   cd frontend
   [ -f .env ] || cp .env.example .env
   npm ci
   npm run dev
   ```

4. Open http://127.0.0.1:5173.

### API configuration and secrets

Set these in `frontend/.env` (separate from the root backend `.env`):

- `VITE_API_BASE_URL` is the browser's API base path, default `/api`. Vite embeds
  `VITE_` values in the client bundle; changing them requires restarting the
  dev server or rebuilding the deployed frontend.
- `RISKFORGE_API_TARGET` is the development proxy's upstream, default
  `http://127.0.0.1:8000`. The proxy strips `/api` before forwarding. It does not
  configure production routing or change the browser's API base URL.

Never put reviewer keys or other secrets in a `VITE_` variable. Configure
`RISKFORGE_REVIEWER_API_KEY` only in the backend environment; enter the reviewer
key in the dashboard at runtime, where it is held only in component memory.

### Frontend checks

After `npm ci`, run from `frontend/`:

```bash
npm run typecheck
npm test
npm run build
```

`npm test` runs Vitest once; `npm run test:watch` is available for local work.
CI runs `npm ci`, `npm test` (Vitest) and `npm run build`, which type-checks
with `tsc --noEmit` before the Vite production build. These tests mock API calls;
they are separate from the browser-to-database suite below.

### Browser review E2E

Playwright drives Chromium through Vite's proxy to real FastAPI and PostgreSQL,
without API mocking. It covers missing/wrong/correct reviewer credentials, queue
loading and details, credential clearing on reload, approve/reject with database
assertions, and a stale second browser receiving 409. The entire original
decision and single review event are checked after resolution.

Use a **new disposable local** database named exactly `riskforge_e2e`. Activate
the repository's Python 3.12 environment first; `python` must resolve to it.
Never repurpose an existing database or point this command at shared infrastructure.
From the repository root, after installing Python dependencies:

```bash
source .venv/bin/activate
export DATABASE_URL=postgresql+psycopg://localhost:5432/riskforge_e2e
createdb -h localhost riskforge_e2e
alembic upgrade head
cd frontend
npm ci
npx playwright install --no-shell chromium
npm run typecheck:e2e
npm run test:e2e
```

The example uses the current local PostgreSQL role; adjust it for your isolated
server. The fixture refuses any other database name or non-loopback host, and
refuses cleanup when non-`E2E-` decisions exist. `E2E-` IDs are reserved for
these disposable tests. Each test resets/seeds two explicitly synthetic REVIEW
decisions and cleans its own data in teardown. Run only one suite per database.
If forcibly interrupted, rerun against the same disposable database to reset it.

The runner starts/stops its own API on 8010 and Vite on 5174; both ports must be
free. It refuses to reuse existing servers. No dataset or serving artifact is
needed: scoring is unavailable without an artifact, but review operations work.
It generates an ephemeral reviewer key, or uses credentials already exported
from a local `.env`; never print those values. Credentials are not traced,
video-recorded or screenshotted. Browser outputs are ignored by Git.

GitHub Actions runs this suite in a separate PostgreSQL 17 service and installs
Chromium with `--with-deps --no-shell`. Tests use one worker and zero retries.
This verifies the review integration, not model accuracy, independent human code
review, screen-reader acceptance or production readiness.

## Current scope

- Reads `GET /health` and displays artifact/database readiness.
- Sends the existing `POST /v1/risk/score` request contract with a stable
  per-submission idempotency key, without reimplementing feature engineering,
  calibration, TreeSHAP or policy thresholds in JavaScript.
- Displays the calibrated probability, original policy decision, model signals,
  risk-reducing signals and model/feature/policy versions returned by the API.
- Reads persisted decisions from the paginated `GET /v1/audit/decisions` endpoint.
- Provides an authenticated review queue and resolution actions. The reviewer key
  is entered at runtime, held only in component memory, and sent in the
  `X-Reviewer-API-Key` header.
- Keeps transaction history input explicit: the caller supplies strictly-prior
  history as JSON, matching the backend's point-in-time contract.

The review workflow is a single configured local reviewer, not a production
identity provider. The original model decision is preserved as REVIEW; human
resolution is stored separately. The dashboard does not fabricate transaction
counts or sample risk decisions.

## Production note

The development proxy is not a production reverse proxy. A deployed frontend
must be served behind a same-origin gateway/reverse proxy or use a deliberately
configured backend CORS allowlist. Do not expose the API with wildcard CORS.
