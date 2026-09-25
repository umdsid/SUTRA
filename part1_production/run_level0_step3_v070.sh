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

echo "Running STRATA v0.7.0 Step 3 tests..."
"$PY" -m pytest -q \
  tests/test_hierarchy_step3_cells.py \
  tests/test_hierarchy_step3_interfaces.py \
  tests/test_hierarchy_step3_patch_retention.py

echo
"$PY" -m sutra.cli.level0_materialization_v070_step3 \
  --project-root "$ROOT" \
  --workers "${SUTRA_LEVEL0_WORKERS:-3}"
