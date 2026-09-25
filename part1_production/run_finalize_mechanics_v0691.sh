#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${STRATA_PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -m sutra.cli.finalize_production_mechanics_v0691 \
  --project-root "$ROOT"
