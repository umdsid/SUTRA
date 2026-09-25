#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${STRATA_PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1

"$PY" -m sutra.cli.native_production_observable_v069 \
  --project-root "$ROOT" \
  --workers "${STRATA_PROD_MECH_WORKERS:-8}" \
  --patch-size "${STRATA_MECH_PATCH_SIZE:-300}" \
  --halo "${STRATA_MECH_HALO:-1}" \
  --atol "${STRATA_PROD_MECH_ATOL:-1e-12}" \
  --btol "${STRATA_PROD_MECH_BTOL:-1e-12}" \
  --solver-agreement-tol "${STRATA_PROD_MECH_AGREE_TOL:-1e-7}" \
  --maxiter-factor "${STRATA_PROD_MECH_MAXITER_FACTOR:-12}"
