#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
unset PYTHONPATH
export PYTHONPATH="$ROOT/src"
"$ROOT/.venv/bin/python" - <<'PYOWN'
import sutra
from pathlib import Path
p = Path(sutra.__file__).resolve()
root = Path.cwd().resolve()
if root not in p.parents:
    raise SystemExit(f"ERROR: SUTRA import ownership failure: {p}")
print(f"SUTRA import ownership: {p}")
PYOWN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
exec "$ROOT/.venv/bin/python" -m sutra.cli.contextual_flow_v0911 \
  --project-root "$ROOT" \
  --workers "${SUTRA_HIERARCHY_WORKERS:-3}" \
  "$@"
