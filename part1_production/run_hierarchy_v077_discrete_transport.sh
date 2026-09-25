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

echo "Running STRATA v0.7.7 discrete-transport tests..."
"$PY" -m pytest -q \
  tests/test_v077_rotation_maps_direction.py \
  tests/test_v077_rotation_inverse.py \
  tests/test_v077_g_isometry.py \
  tests/test_v077_one_sided_unresolved.py \
  tests/test_v077_zero_zero_identity.py \
  tests/test_v077_directed_geodesic_triangle.py

echo
"$PY" -m sutra.cli.discrete_transport_v077 \
  --project-root "$ROOT" \
  --landmarks "${STRATA_V077_LANDMARKS:-16}" \
  --audit-edges "${STRATA_V077_AUDIT_EDGES:-256}"
