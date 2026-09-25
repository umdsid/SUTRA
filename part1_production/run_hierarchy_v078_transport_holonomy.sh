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

echo "Running STRATA v0.7.8 transport-holonomy tests..."
"$PY" -m pytest -q \
  tests/test_v078_triangle_enumeration.py \
  tests/test_v078_chordless_quad_enumeration.py \
  tests/test_v078_identity_loop_zero.py \
  tests/test_v078_3d_triangle_nontrivial.py \
  tests/test_v078_loop_reversal_inverse.py \
  tests/test_v078_coordinate_invariance.py \
  tests/test_v078_unresolved_loop_not_repaired.py

echo
"$PY" -m sutra.cli.transport_holonomy_v078 \
  --project-root "$ROOT" \
  --max-quads "${STRATA_V078_MAX_QUADS:-20000}" \
  --coordinate-checks "${STRATA_V078_COORD_CHECKS:-32}"
