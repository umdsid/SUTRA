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

echo "Running STRATA v0.7.4.3 channel-direction tests..."
"$PY" -m pytest -q \
  tests/test_v0743_channel_cancellation.py \
  tests/test_v0743_directional_content_bound.py \
  tests/test_v0743_channel_reversal.py \
  tests/test_v0743_no_false_direction.py

echo
"$PY" -m sutra.cli.channel_direction_audit_v0743 \
  --project-root "$ROOT" \
  --workers "${STRATA_V0743_WORKERS:-3}" \
  --comm-support-positive-quantile "${STRATA_V0743_COMM_SUPPORT_Q:-0.10}"
