#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-https://api.fwvgoldmindai.com}"
MAX_RETRIES="${MAX_RETRIES:-15}"
SLEEP_SECS="${SLEEP_SECS:-5}"

echo "🔎 Running smoke tests against: ${BASE_URL}"

endpoints=(
  "/health"
  "/api/insights/macro"
  "/api/insights/midlayer"
  "/api/summary"
  "/api/v1/alerts"
)

retry_curl() {
  local url="$1"
  local attempt=1
  local http_code
  while (( attempt <= MAX_RETRIES )); do
    http_code=$(curl -sS -o /tmp/resp.json -w "%{http_code}" "${url}" || echo "000")
    if [[ "${http_code}" == "200" ]]; then
      echo "✅ 200 OK ${url}"
      # Show a tiny snippet to prove it's real JSON
      head -c 200 /tmp/resp.json || true
      echo -e "\n----"
      return 0
    fi
    echo "⏳ (${attempt}/${MAX_RETRIES}) ${url} -> HTTP ${http_code}; retrying in ${SLEEP_SECS}s..."
    sleep "${SLEEP_SECS}"
    ((attempt++))
  done
  echo "❌ FAILED: ${url} did not return 200 after ${MAX_RETRIES} attempts"
  echo "Last body:"
  cat /tmp/resp.json || true
  return 1
}

overall=0
for ep in "${endpoints[@]}"; do
  if ! retry_curl "${BASE_URL}${ep}"; then
    overall=1
  fi
done

if [[ "${overall}" -ne 0 ]]; then
  echo "❌ Smoke tests FAILED"
  exit 1
fi

echo "🎉 All smoke tests PASSED"
