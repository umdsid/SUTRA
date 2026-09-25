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

echo "STRATA 0.6.4 | Mechanics corrections + rank recovery"
echo "Outputs: results/corrections/v064/"
echo "Frozen biological geometry is read-only."
echo

"$PY" -m sutra.cli.native_corrections_v064 \
  --project-root "$ROOT" \
  --workers "${STRATA_CORRECTION_WORKERS:-3}" \
  --patch-size "${STRATA_MECH_PATCH_SIZE:-300}" \
  --halo "${STRATA_MECH_HALO:-1}" \
  --unstable-area-max "${STRATA_MICROGAP_MAX_AREA:-64}" \
  --unstable-survival-max "${STRATA_MICROGAP_MAX_SURVIVAL:-0.10}"
