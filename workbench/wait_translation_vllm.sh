#!/usr/bin/env bash
set -euo pipefail
CURL_BIN="${CURL_BIN:-/home/aim0/anaconda3/bin/curl}"
for attempt in $(seq 1 300); do
  if "${CURL_BIN}" --silent --show-error --fail http://127.0.0.1:8090/health >/dev/null 2>&1; then
    exit 0
  fi
  sleep 1
done
echo "Index-Translate vLLM server did not become healthy within 300 seconds." >&2
exit 1
