# GoldMIND v11.2 Integration Audit

## Baseline

- Repository: `fosterwealthventures/goldmind-ai`
- Base branch: `main`
- Recovery branch: `recovery/goldmind-v11-2`
- Base commit: `27f4b68a766444b1d011620667d6462a25fddfff`

## Uploaded ZIP structure confirmed

The uploaded `goldmind-ai-main.zip` contains a reusable Vite/React frontend plus Python API/compute services and deployment infrastructure.

Reusable frontend paths:

- `frontend/web/src/App.jsx`
- `frontend/web/src/api.js`
- `frontend/web/src/apiClient.js`
- `frontend/web/src/computeClient.js`
- `frontend/web/src/panels/BiasCard.jsx`
- `frontend/web/src/panels/PredictCard.jsx`
- `frontend/web/src/panels/InsightsCard.jsx`

Reusable backend/infrastructure paths:

- `api/`
- `compute/app/server.py`
- `compute/engine/`
- `cloudbuild.yaml`
- `deploy-compute.sh`
- `scripts/`
- Docker and Cloud Run configuration

Unsafe behavior to replace or disable:

- synthetic/mock history used as production fallback
- prediction guidance produced from fabricated candles
- any `LIVE` state shown while the source is mock
- old prediction output acting as the authoritative permission decision

## Compile audit completed

Each recovered JSX file was copied into the ZIP's existing React 19 / Vite 7 frontend and compiled independently with `npm ci` and `npm run build`.

All four production builds completed successfully:

| Recovered file | Build result |
|---|---|
| `GoldMIND-v11_2-preview.jsx` | Passed |
| `GoldMIND-v11.2.jsx` | Passed |
| `GoldMIND-v11.2-engine.jsx` | Passed |
| `GoldMIND-Complete.jsx` | Passed |

This confirms the recovered files are valid JSX and are compatible with the ZIP's current React/Vite frontend. The earlier suspected syntax corruption was a file-viewer rendering issue, not corruption in the uploaded originals.

## Integration assignment

- `GoldMIND-v11.2-engine.jsx`: authoritative intelligence and trade-permission layer
- `GoldMIND-v11.2.jsx`: production Setup/Home decision interface
- `GoldMIND-v11_2-preview.jsx`: development-only scenario route
- `GoldMIND-Complete.jsx`: source for chart, MT5 account, journal, and analytics components
- Existing ZIP/GitHub app: API, compute service, routing, deployment, and legacy fallback during migration

## Required safety behavior

Production must return a paused/unavailable state when live market data cannot be obtained. Mock scenarios remain allowed only on the clearly labeled preview route.

## Next implementation checkpoint

1. Preserve recovered originals under `frontend/web/src/recovery/`.
2. Add a preview route without replacing the legacy app.
3. Normalize the existing API candle/tick response.
4. Connect the normalized feed to the v11.2 engine.
5. Use `calcPermission()` as the sole authority for trade/wait/stop verdicts.
6. Mount the v11.2 presentation layer over the engine output.
7. Extract Chart, MT5, Journal, and Analytics panels from `GoldMIND-Complete.jsx`.
