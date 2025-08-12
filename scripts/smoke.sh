#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/smoke.sh                      # discovers URLs with gcloud
#   ./scripts/smoke.sh <API_BASE> <COMPUTE_BASE>
#
# Env (for discovery):
#   GCP_PROJECT_ID, GCP_REGION
#   GCP_API_SERVICE=goldmind-api (optional)
#   GCP_COMPUTE_SERVICE=goldmind-compute (optional)
#   INTERNAL_SHARED_SECRET=... (optional; if set, tests compute/predict)

API_BASE="${1:-}"
COMPUTE_BASE="${2:-}"

pass(){ echo "✅ $1"; }
fail(){ echo "❌ $1"; exit 1; }
code_of(){ curl -s -o /dev/null -w "%{http_code}" "$@"; }

discover_url () {
  local svc="$1"
  gcloud run services describe "$svc" \
    --project="${GCP_PROJECT_ID}" \
    --region="${GCP_REGION}" \
    --format='value(status.url)'
}

if [[ -z "$API_BASE" ]]; then
  : "${GCP_PROJECT_ID:?set GCP_PROJECT_ID}"; : "${GCP_REGION:?set GCP_REGION}"
  API_BASE="$(discover_url "${GCP_API_SERVICE:-goldmind-api}")"
fi

if [[ -z "$COMPUTE_BASE" ]]; then
  : "${GCP_PROJECT_ID:?set GCP_PROJECT_ID}"; : "${GCP_REGION:?set GCP_REGION}"
  COMPUTE_BASE="$(discover_url "${GCP_COMPUTE_SERVICE:-goldmind-compute}")"
fi

echo "API_BASE     → $API_BASE"
echo "COMPUTE_BASE → $COMPUTE_BASE"

# ---------- API health ----------
c=$(code_of "$API_BASE/health")
[[ "$c" == "200" ]] && pass "API /health 200" || fail "API /health $c"

# ---------- API version (optional) ----------
c=$(code_of "$API_BASE/version")
[[ "$c" == "200" ]] && pass "API /version 200" || echo "ℹ️  API /version not present ($c)"

# ---------- API predict (forwards to compute) ----------
c=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"XAU","user":{"style":"Day Trading","capital":1000,"daily_target":80},"options":{"seq_len":60}}' \
  "$API_BASE/predict")
[[ "$c" == "200" ]] && pass "API /predict 200" || fail "API /predict $c"

# ---------- API feedback ----------
c=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d '{"user":"demo","feedback":"works"}' \
  "$API_BASE/feedback")
[[ "$c" == "200" ]] && pass "API /feedback 200" || fail "API /feedback $c"

# ---------- Compute health ----------
c=$(code_of "$COMPUTE_BASE/health")
[[ "$c" == "200" ]] && pass "Compute /health 200" || fail "Compute /health $c"

# ---------- (Optional) direct compute predict (requires secret) ----------
if [[ -n "${INTERNAL_SHARED_SECRET:-}" ]]; then
  c=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -H "X-Internal-Secret: ${INTERNAL_SHARED_SECRET}" \
    -d '{"symbol":"XAU","user":{"style":"Day Trading"},"options":{"seq_len":30}}' \
    "$COMPUTE_BASE/compute/predict")
  [[ "$c" == "200" ]] && pass "Compute /compute/predict 200 (with secret)" || fail "Compute /compute/predict $c"
else
  echo "ℹ️  Skipping direct compute/predict (set INTERNAL_SHARED_SECRET to test)"
fi

pass "Smoke tests passed"
