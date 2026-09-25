#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Prevent parent BLAS libraries from creating thread teams before workers spawn.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1

echo "STRATA 0.6.1 | Accelerated native boundary-aware mechanics"
echo "Geometry: vectorized + persistent SHA256 cache"
echo "Mechanics: global patch-process pool"
echo

"$PY" -m sutra.cli.native_mechanics_fast \
  --project-root "$ROOT" \
  --patch-size "${STRATA_MECH_PATCH_SIZE:-300}" \
  --halo "${STRATA_MECH_HALO:-1}" \
  --geometry-workers "${STRATA_GEOM_WORKERS:-3}" \
  --patch-workers "${STRATA_PATCH_WORKERS:-8}"
