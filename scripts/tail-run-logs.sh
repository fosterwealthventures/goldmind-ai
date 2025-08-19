#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/tail-run-logs.sh <service> [region] [--no-follow] [--freshness=5m] [--limit=200]

Examples:
  ./scripts/tail-run-logs.sh goldmind-api us-central1
  ./scripts/tail-run-logs.sh goldmind-compute us-central1 --no-follow --freshness=10m --limit=300
EOF
}

SERVICE="${1:-}"
if [[ -z "${SERVICE}" || "${SERVICE}" == "-h" || "${SERVICE}" == "--help" ]]; then
  usage; exit 1
fi

REGION="${2:-us-central1}"
FOLLOW=1
FRESHNESS="5m"
LIMIT="200"

# optional flags start at $3
for arg in "${@:3}"; do
  case "$arg" in
    --no-follow)   FOLLOW=0 ;;
    --freshness=*) FRESHNESS="${arg#*=}" ;;
    --limit=*)     LIMIT="${arg#*=}" ;;
    *) echo "Unknown option: $arg" >&2; usage; exit 2 ;;
  esac
done

# require gcloud
if ! command -v gcloud >/dev/null 2>&1; then
  echo "❌ gcloud not found in PATH" >&2
  exit 3
fi

echo "Region : ${REGION}"
echo "Service: ${SERVICE}"

REV="$(gcloud run services describe "${SERVICE}" --region "${REGION}" \
  --format='value(status.latestReadyRevisionName)')"

if [[ -z "${REV}" ]]; then
  echo "❌ Latest revision is empty. Check service/region." >&2
  exit 4
fi

echo "Revision: ${REV}"

# Properly quoted Logging filter for Cloud Run revisions
FILTER="resource.type=\"cloud_run_revision\" \
AND resource.labels.service_name=\"${SERVICE}\" \
AND resource.labels.location=\"${REGION}\" \
AND resource.labels.revision_name=\"${REV}\""

echo "Filter : ${FILTER}"
echo

# Prefer streaming tail if available; otherwise read once (or poll if follow requested)
if gcloud logging tail --help >/dev/null 2>&1; then
  if [[ "${FOLLOW}" -eq 1 ]]; then
    echo "▶ Streaming logs (Ctrl+C to stop)..."
    exec gcloud logging tail --log-filter="${FILTER}" \
      --format='table(timestamp, severity, resource.labels.revision_name, textPayload, jsonPayload, httpRequest.status)'
  else
    echo "▶ Reading last ${LIMIT} entries (freshness=${FRESHNESS})..."
    exec gcloud logging read "${FILTER}" --freshness="${FRESHNESS}" --limit="${LIMIT}" --order=desc \
      --format='table(timestamp, severity, resource.labels.revision_name, textPayload, jsonPayload, httpRequest.status)'
  fi
else
  echo "⚠ No gcloud logging tail; using read loop instead."
  while true; do
    gcloud logging read "${FILTER}" --freshness="${FRESHNESS}" --limit="${LIMIT}" --order=desc \
      --format='table(timestamp, severity, resource.labels.revision_name, textPayload, jsonPayload, httpRequest.status)' || true
    [[ "${FOLLOW}" -eq 1 ]] || break
    sleep 2
  done
fi
