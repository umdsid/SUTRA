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

echo "Running STRATA v0.7.6.1 tests..."
"$PY" -m pytest -q \
  tests/test_v0761_exact_gradient_zero_residual.py \
  tests/test_v0761_cycle_inconsistency_detected.py \
  tests/test_v0761_tail_threshold.py \
  tests/test_v0761_level1_enrichment.py

echo
"$PY" -m sutra.cli.pressure_tail_audit_v0761 --project-root "$ROOT"
