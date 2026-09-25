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

echo "Running STRATA v0.7.6.2 tests..."
"$PY" -m pytest -q \
  tests/test_v0762_exact_decomposition.py \
  tests/test_v0762_exact_gradient_zero_residual.py \
  tests/test_v0762_inconsistent_cycle_residual.py \
  tests/test_v0762_gauge_invariance.py \
  tests/test_v0762_consistency_ratio_not_gate.py

echo
"$PY" -m sutra.cli.pressure_field_finalize_v0762 --project-root "$ROOT"
