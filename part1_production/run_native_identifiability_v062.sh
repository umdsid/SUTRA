#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
echo "STRATA 0.6.2 | Mechanical identifiability decomposition"
"$PY" -m sutra.cli.native_identifiability_v062   --project-root "$ROOT"   --patch-size "${STRATA_MECH_PATCH_SIZE:-300}"   --halo "${STRATA_MECH_HALO:-1}"   --workers "${STRATA_IDENT_WORKERS:-8}"   --small-gap-pixels "${STRATA_SMALL_GAP_PIXELS:-64}"   --medium-gap-pixels "${STRATA_MEDIUM_GAP_PIXELS:-256}"
