#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/run_missing_stage.py" --project "$ROOT" --stage "h071"
