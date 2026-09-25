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

"$PY" -m sutra.cli.native_junction_recovery_v065 \
  --project-root "$ROOT" \
  --workers "${STRATA_JUNCTION_WORKERS:-3}" \
  --patch-size "${STRATA_MECH_PATCH_SIZE:-300}" \
  --halo "${STRATA_MECH_HALO:-1}" \
  --local-pad "${STRATA_JUNCTION_LOCAL_PAD:-5}"
