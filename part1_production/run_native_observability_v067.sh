#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1

echo "STRATA 0.6.7 | Variable-level mechanical observability"
echo "Representation: persistent boundaries + accepted v0.6.5 recovered junctions"
echo "No regularization of unresolved patches."
echo

"$PY" -m sutra.cli.native_observability_v067 \
  --project-root "$ROOT" \
  --workers "${STRATA_OBS_WORKERS:-8}" \
  --patch-size "${STRATA_MECH_PATCH_SIZE:-300}" \
  --halo "${STRATA_MECH_HALO:-1}" \
  --dense-nvar-cap "${STRATA_OBS_DENSE_CAP:-450}" \
  --dense-nullity-cap "${STRATA_OBS_NULLITY_CAP:-40}" \
  --numerical-rel-tol "${STRATA_OBS_TOL:-1e-9}"
