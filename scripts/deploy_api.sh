#!/usr/bin/env bash
set -Eeuo pipefail

# ----------- Inputs / Defaults (from Cloud Build substitutions) -----------
REGION="${_REGION:-us-central1}"
REPO="${_REPO:-goldmind}"
SERVICE_API="${_SERVICE_API:-goldmind-api}"
SERVICE_COMPUTE="${_SERVICE_COMPUTE:-goldmind-compute}"
SECRET_NAME="${_SECRET_NAME:-internal-shared-secret}"  # Secret Manager name

# PROJECT_ID may not be exported by Cloud Build. Resolve robustly:
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "PROJECT_ID not found. Pass it via env or set gcloud config."
  exit 1
fi

# BUILD_ID must match the tag used during docker build/push
if [[ -z "${BUILD_ID:-}" ]]; then
  echo "BUILD_ID not provided to deploy step. Make sure cloudbuild.yaml passes it via env."
  exit 1
fi

# ----------- Helpers -----------
on_err() {
  echo "❌ Deploy failed. Recent Cloud Run logs (last 2h):"
  gcloud logging read \
    'resource.type="cloud_run_revision" AND resource.labels.location="'"${REGION}"'"' \
    --project "${PROJECT_ID}" --limit 200 --freshness=2h \
    --format='value(timestamp, severity, textPayload)' || true
}
trap on_err ERR

health_wait() {
  local url="$1"
  local name="$2"
  echo "⏳ Waiting for ${name} to be healthy at ${url}/health ..."
  local i=0
  until curl -fsS --connect-timeout 3 "${url}/health" >/dev/null 2>&1; do
    i=$((i+1))
    if [[ "$i" -gt 120 ]]; then
      echo "❌ ${name} health check timed out."
      exit 1
    fi
    printf "  (%s/120) not ready yet...\n" "$i"
    sleep 5
  done
  echo "✅ ${name} healthy."
}

# ----------- Deploy COMPUTE first -----------
echo "▶️  Deploying ${SERVICE_COMPUTE} ..."
gcloud run deploy "${SERVICE_COMPUTE}" \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/compute:${BUILD_ID}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --quiet \
  --update-secrets="INTERNAL_SHARED_SECRET=${SECRET_NAME}:latest" \
  --update-env-vars="ENV=prod,APP_VERSION=${BUILD_ID}"

COMPUTE_URL="$(gcloud run services describe "${SERVICE_COMPUTE}" \
  --region="${REGION}" --format='value(status.url)')"

health_wait "${COMPUTE_URL}" "${SERVICE_COMPUTE}"

# ----------- Deploy API (points at COMPUTE) -----------
echo "▶️  Deploying ${SERVICE_API} ..."

# Use an env file for non-secret vars so commas/URLs are safe
ENV_FILE="$(mktemp)"
cat > "${ENV_FILE}" <<EOF
ENV: "prod"
APP_VERSION: "${BUILD_ID}"
COMPUTE_URL: "${COMPUTE_URL}"
CORS_ALLOW_ORIGINS: "https://fwvgoldmindai.com,https://www.fwvgoldmindai.com"
EOF

gcloud run deploy "${SERVICE_API}" \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/api:${BUILD_ID}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --quiet \
  --env-vars-file="${ENV_FILE}" \
  --update-secrets="INTERNAL_SHARED_SECRET=${SECRET_NAME}:latest"

API_URL="$(gcloud run services describe "${SERVICE_API}" \
  --region="${REGION}" --format='value(status.url)')"

health_wait "${API_URL}" "${SERVICE_API}"

echo "🎉 Deploy complete."
echo "   Compute: ${COMPUTE_URL}"
echo "   API:     ${API_URL}"
