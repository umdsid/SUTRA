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

echo "Running STRATA v0.7.4 tests..."
"$PY" -m pytest -q \
  tests/test_v074_effective_expression.py \
  tests/test_v074_effective_gate.py \
  tests/test_v074_maximal_matching.py

echo
"$PY" -m sutra.cli.hierarchy_effective_flow_v074 \
  --project-root "$ROOT" \
  --workers "${STRATA_V074_WORKERS:-3}" \
  --max-levels "${STRATA_V074_MAX_LEVELS:-250}" \
  --checkpoint-every "${STRATA_V074_CHECKPOINT_EVERY:-2}"
