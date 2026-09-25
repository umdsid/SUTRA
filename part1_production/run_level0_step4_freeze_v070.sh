#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${STRATA_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python; fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running STRATA v0.7.0 Step 4 freeze-audit tests..."
"$PY" -m pytest -q \
  tests/test_level0_freeze_exact_copy.py \
  tests/test_level0_freeze_retention.py \
  tests/test_level0_freeze_csr.py \
  tests/test_level0_freeze_digest.py

echo
"$PY" -m sutra.cli.level0_freeze_audit_v070 \
  --project-root "$ROOT" \
  --workers "${STRATA_LEVEL0_AUDIT_WORKERS:-3}"
