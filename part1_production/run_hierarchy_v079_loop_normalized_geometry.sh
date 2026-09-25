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

echo "Running STRATA v0.7.9 loop-normalized geometry tests..."
"$PY" -m pytest -q \
  tests/test_v079_triangle_area.py \
  tests/test_v079_density_definition.py \
  tests/test_v079_degenerate_density_undefined.py \
  tests/test_v079_identity_density_zero.py \
  tests/test_v079_cell_aggregation.py \
  tests/test_v079_coordinate_resolver.py

echo
"$PY" -m sutra.cli.discrete_curvature_summary_v079 \
  --project-root "$ROOT"
