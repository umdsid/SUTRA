#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m sutra.cli.tranche2_2e_radius_calibration \
  --project-root "$ROOT" \
  --workers "${STRATA_RADIUS_WORKERS:-3}"
