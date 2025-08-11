#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-https://api.fwvgoldmindai.com}"

pass(){ echo "✅ $1"; }
fail(){ echo "❌ $1"; exit 1; }

# ---------- Health ----------
code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/health")
if [[ "$code" != "200" ]]; then
  code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/healthz")
fi
[[ "$code" == "200" ]] && pass "health 200" || fail "health $code"

# ---------- Predict ----------
code=$(curl -s -o /dev/null -w "%{http_code}" -H "Content-Type: application/json" \
  -d '{"input":[2000,2010,2025,2035]}' "$BASE/predict")
[[ "$code" == "200" ]] && pass "/predict 200" || fail "/predict $code"

# ---------- Analyze ----------
code=$(curl -s -o /dev/null -w "%{http_code}" -H "Content-Type: application/json" \
  -d '{"text":"I think BUY makes sense","user_id":"demo"}' "$BASE/analyze/text")
[[ "$code" == "200" ]] && pass "/analyze/text 200" || fail "/analyze/text $code"

# ---------- Resolve (also used to produce a trace id) ----------
RESOLVE_JSON=$(curl -s -H "Content-Type: application/json" \
  -d '{"input":[2000,2010,2025],"text":"buy breakout","user_id":"demo","trading_style":"Day Trading","investment_amount":1000,"time_frame":"1h"}' \
  "$BASE/resolve")
code=$(jq -r '."id" // empty' <<<"$RESOLVE_JSON" >/dev/null 2>&1; echo $?)
if [[ "$code" -ne 0 ]]; then
  # jq not available: fallback to sed
  REC_ID=$(echo "$RESOLVE_JSON" | sed -n 's|.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*|\1|p')
else
  REC_ID=$(jq -r '.id // empty' <<<"$RESOLVE_JSON")
fi

if [[ -z "${REC_ID:-}" ]]; then
  echo "$RESOLVE_JSON"
  fail "/resolve did not return id"
else
  pass "/resolve 200 (id: $REC_ID)"
fi

# ---------- Trace (GET /trace/:id) ----------
code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/trace/$REC_ID")
[[ "$code" == "200" ]] && pass "/trace/:id 200" || fail "/trace/:id $code"

# ---------- Trace (POST /trace) ----------
code=$(curl -s -o /dev/null -w "%{http_code}" -H "Content-Type: application/json" \
  -d "{\"id\":\"$REC_ID\"}" "$BASE/trace")
[[ "$code" == "200" ]] && pass "/trace POST 200" || fail "/trace POST $code"

# ---------- Feedback ----------
code=$(curl -s -o /dev/null -w "%{http_code}" -H "Content-Type: application/json" \
  -d '{"user_id":"demo","trading_style":"Day Trading","investment_amount":1000,"followed_recommendation":"yes","feedback_text":"worked well"}' \
  "$BASE/feedback")
[[ "$code" == "200" ]] && pass "/feedback 200" || fail "/feedback $code"

# ---------- Settings PUT ----------
put_code=$(curl -s -o /dev/null -w "%{http_code}" -H "Content-Type: application/json" \
  -X PUT -d '{"smooth_lines":false,"default_trading_style":"Swing Trading","default_time_frame":"4h"}' \
  "$BASE/settings")

[[ "$put_code" == "200" ]] && pass "/settings PUT 200" || fail "/settings PUT $put_code"



