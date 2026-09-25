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

echo "Running STRATA v0.7.3 tests..."
"$PY" -m pytest -q \
  tests/test_v073_maximal_matching.py \
  tests/test_v073_stop_conditions.py \
  tests/test_v073_failure_counts.py

ARGS=(
  --project-root "$ROOT"
  --workers "${STRATA_V073_WORKERS:-3}"
  --max-levels "${STRATA_V073_MAX_LEVELS:-250}"
  --checkpoint-every "${STRATA_V073_CHECKPOINT_EVERY:-5}"
)

if [[ "${STRATA_V073_RESUME:-0}" == "1" ]]; then
  ARGS+=(--resume)
fi

echo
"$PY" -m sutra.cli.hierarchy_full_v073 "${ARGS[@]}"
