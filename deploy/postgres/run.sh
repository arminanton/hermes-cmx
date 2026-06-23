#!/usr/bin/env bash
# Bring up the cmx-postgres backend (Postgres 17 + pgvector + pg_trgm) for hermes-cmx.
# Idempotent: safe to re-run. Sidesteps host glibc — everything ships in the container.
set -euo pipefail
cd "$(dirname "$0")"

ENGINE="$(command -v podman >/dev/null 2>&1 && echo podman || echo docker)"
echo ">> using container engine: $ENGINE"

if $ENGINE compose version >/dev/null 2>&1; then
  $ENGINE compose -f compose.yaml up -d
else
  # older podman: use the external podman-compose provider
  $ENGINE-compose -f compose.yaml up -d
fi

echo ">> waiting for healthy..."
for i in $(seq 1 30); do
  status="$($ENGINE inspect -f '{{.State.Health.Status}}' cmx-postgres 2>/dev/null || echo starting)"
  [ "$status" = "healthy" ] && { echo ">> cmx-postgres healthy on host port 5433"; break; }
  sleep 2
done

echo ">> extensions:"
$ENGINE exec cmx-postgres psql -U cmx -d cmx -c "\dx" | grep -E "pg_trgm|vector" || true
echo ">> DSN:  host=127.0.0.1 port=5433 dbname=cmx user=cmx password=cmx_local_dev"
echo ">> verify: python3 spike_verify.py"
