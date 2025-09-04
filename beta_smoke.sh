#!/usr/bin/env bash
set -euo pipefail

REGION=us-central1
API_SERVICE=goldmind-api
MAPPED=https://api.fwvgoldmindai.com

API_URL=$(gcloud run services describe "$API_SERVICE" --region="$REGION" --format='value(status.url)')
echo "API_URL=$API_URL"
echo "MAPPED =$MAPPED"
echo

step() { echo -e "\n=== $* ==="; }

step "Health (direct)"
curl -sS -f "$API_URL/health" | jq . >/dev/null && echo OK

step "Health (mapped domain)"
curl -sS -f "$MAPPED/health" | jq . >/dev/null && echo OK

step "CORS preflight /predict from fwvgoldmindai.com"
curl -sS -f -i -X OPTIONS "$MAPPED/predict" \
  -H 'Origin: https://fwvgoldmindai.com' \
  -H 'Access-Control-Request-Method: POST' \
  -H 'Access-Control-Request-Headers: Content-Type' \
  | sed -n '1,20p'

step "GET /settings"
curl -sS -f "$MAPPED/settings" | jq . >/dev/null && echo OK

step "PUT /settings"
curl -sS -f -X PUT "$MAPPED/settings" -H 'Content-Type: application/json' \
  -d '{"ui_theme":"dark","notify":false}' | jq . >/dev/null && echo OK

step "POST /predict"
curl -sS -f -X POST "$MAPPED/predict" -H 'Content-Type: application/json' \
  -d '{"symbol":"XAUUSD","horizon":"1h","amount":1000,"style":"swing"}' | jq . >/dev/null && echo OK

echo -e "\n🎉 All beta smoke checks passed"
