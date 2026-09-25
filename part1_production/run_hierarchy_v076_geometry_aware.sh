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

echo "Running STRATA v0.7.6 adversarial tests..."
"$PY" -m pytest -q \
  tests/test_v076_pair_cost_orientation.py \
  tests/test_v076_direction_retained.py \
  tests/test_v076_covector_aggregation_bound.py \
  tests/test_v076_geometry_matching_disjoint.py \
  tests/test_v076_pressure_reconstruction.py

echo
"$PY" -m sutra.cli.hierarchy_geometry_aware_v076 \
  --project-root "$ROOT" \
  --workers "${STRATA_V076_WORKERS:-3}" \
  --max-levels "${STRATA_V076_MAX_LEVELS:-250}"
