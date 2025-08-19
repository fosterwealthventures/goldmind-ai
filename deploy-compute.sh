#!/usr/bin/env bash
set -euo pipefail

# ===== Fixed values for your project/service =====
PROJECT_ID="goldmind-api"
REGION="us-central1"
COMPUTE_SERVICE="goldmind-compute"
IMAGE="gcr.io/${PROJECT_ID}/${COMPUTE_SERVICE}:latest"

echo ">>> Using project: ${PROJECT_ID}, region: ${REGION}, service: ${COMPUTE_SERVICE}"

# ===== gcloud setup =====
gcloud config set project "${PROJECT_ID}" >/dev/null
gcloud auth configure-docker --quiet

# ===== Build new image from compute/ (ensures app.server:app is the entrypoint) =====
echo ">>> Building image ${IMAGE}"
docker build -t "${IMAGE}" -f compute/Dockerfile compute

# ===== Push =====
echo ">>> Pushing image ${IMAGE}"
docker push "${IMAGE}"

# ===== Deploy to Cloud Run =====
# - Escape commas in CORS_ALLOW_ORIGINS with a backslash: \,
# - Clear any previously set INTERNAL_SHARED_SECRET so we don't get 401s
echo ">>> Deploying to Cloud Run"
gcloud run deploy "${COMPUTE_SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars=APP_VERSION=v1.0.0,ENV=prod,CORS_ALLOW_ORIGINS=https://fwvgoldmindai.com\,https://www.fwvgoldmindai.com \
  --clear-env-vars=INTERNAL_SHARED_SECRET

# ===== Show service URL and active revision =====
SERVICE_URL="$(gcloud run services describe "${COMPUTE_SERVICE}" --region "${REGION}" --format='value(status.url)')"
ACTIVE_REV="$(gcloud run services describe "${COMPUTE_SERVICE}" --region "${REGION}" --format='value(status.traffic[0].revisionName)')"
echo "Compute URL → ${SERVICE_URL}"
echo "Active revision → ${ACTIVE_REV}"

# ===== Smoke tests directly against compute =====
set -x
curl -i "${SERVICE_URL}/health"
curl -i "${SERVICE_URL}/version"
curl -i "${SERVICE_URL}/v1/importance?symbol=GLD"
curl -i "${SERVICE_URL}/v1/alerts?symbol=GLD"
curl -i "${SERVICE_URL}/v1/bias?symbol=GLD"
curl -i -X POST "${SERVICE_URL}/compute/predict" -H "Content-Type: application/json" -d '{"symbol":"XAU"}'
set +x

echo "✅ Compute smoke finished."

# ===== API smoke through your mapped domain =====
BASE="https://api.fwvgoldmindai.com"
echo ">>> API smoke via ${BASE}"
set -x
curl -i "${BASE}/health"
curl -i "${BASE}/version"
curl -i "${BASE}/v1/importance?symbol=GLD"
curl -i "${BASE}/v1/alerts?symbol=GLD"
curl -i "${BASE}/v1/bias?symbol=GLD"
curl -i -X POST "${BASE}/predict" -H "Content-Type: application/json" -d '{"symbol":"XAU","user":{"id":"demo"},"options":{"horizon":"1d"}}'
set +x

echo "✅ All smokes finished."
