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

echo "Running STRATA v0.7.2 tests..."
"$PY" -m pytest -q \
  tests/test_v072_conjunctive_gate.py \
  tests/test_v072_matching.py \
  tests/test_v072_contraction.py \
  tests/test_v072_communication.py

echo
"$PY" -m sutra.cli.hierarchy_short_pilot_v072 \
  --project-root "$ROOT" \
  --workers "${STRATA_V072_WORKERS:-3}" \
  --levels "${STRATA_V072_LEVELS:-3}" \
  --max-pair-fraction "${STRATA_V072_PAIR_FRACTION:-0.005}" \
  --max-pairs-per-level "${STRATA_V072_MAX_PAIRS:-250}" \
  --min-mechanics-support-fraction "${STRATA_V072_MIN_MECH_SUPPORT:-0.50}"
