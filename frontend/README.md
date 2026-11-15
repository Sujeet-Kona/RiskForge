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
Current CI typechecks and builds the frontend but **does not run Vitest yet**.
Adding Vitest to the shared pipeline requires separate permission.

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
