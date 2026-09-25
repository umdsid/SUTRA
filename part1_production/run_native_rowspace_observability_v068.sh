#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1

echo "STRATA 0.6.8 | Scalable numerical row-space observability"
echo "One rank-revealing QR per patch; no n-variable or nullity ceiling."
echo "v0.6.7 dense-SVD outputs are the regression standard."
echo

"$PY" -m sutra.cli.native_rowspace_observability_v068 \
  --project-root "$ROOT" \
  --workers "${STRATA_ROWSPACE_WORKERS:-8}" \
  --patch-size "${STRATA_MECH_PATCH_SIZE:-300}" \
  --halo "${STRATA_MECH_HALO:-1}" \
  --rank-rcond "${STRATA_ROWSPACE_RCOND:-1e-9}" \
  --observable-tol "${STRATA_ROWSPACE_OBS_TOL:-1e-8}" \
  --marginal-tol "${STRATA_ROWSPACE_MARGINAL_TOL:-1e-6}" \
  --min-regression-agreement "${STRATA_ROWSPACE_REGRESSION_MIN:-0.99}"
