#!/usr/bin/env bash
set -euo pipefail

API_URL="${1:?usage: ./smoke.sh <api-url>}"
echo "Testing API: $API_URL"

echo "# /health"
curl -fsS "$API_URL/health" | jq .

echo "# /version"
curl -fsS "$API_URL/version" | jq .

echo "# /predict"
curl -fsS -X POST "$API_URL/predict" -H 'Content-Type: application/json' \
  -d '{"symbol":"XAUUSD","horizon":"1d","amount":1000,"style":"day"}' | jq .

echo "# /settings GET"
curl -fsS "$API_URL/settings" | jq .

echo "# /settings PUT"
curl -fsS -X PUT "$API_URL/settings" -H 'Content-Type: application/json' \
  -d '{"risk_mode":"aggressive"}' | jq .
