# RiskForge dashboard

React + TypeScript frontend for the existing RiskForge FastAPI scoring service.

## Local development

1. Start PostgreSQL and apply the existing migration from the repository root.
2. Build the frozen serving artifact using the documented offline pipeline.
3. Start the API from the repository root:

   ```bash
   uvicorn app.main:app --reload
   ```

4. In a second terminal:

   ```bash
   cd frontend
   cp .env.example .env
   npm install
   npm run dev
   ```

5. Open http://127.0.0.1:5173.

The Vite development server proxies `/api` requests to the FastAPI service. Set
`RISKFORGE_API_TARGET` in `.env` if the API runs elsewhere.

## Current scope

- Reads `GET /health` and displays artifact/database readiness.
- Sends the existing `POST /v1/risk/score` request contract without reimplementing
  feature engineering, calibration, TreeSHAP or policy thresholds in JavaScript.
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
