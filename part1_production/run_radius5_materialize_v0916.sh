#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/Desktop/SUTRA}"
ROOT="$(cd "$ROOT" && pwd)"
PY="$ROOT/.venv/bin/python"
VERIFY="$ROOT/scripts/verify_radius5_v0916.py"

if "$PY" "$VERIFY" "$ROOT"; then
  echo "[REUSE] existing radius-5 production geometry is valid; no rematerialization needed"
  exit 0
fi

echo "[BUILD] radius-5 production geometry missing or invalid; materializing..."
"$PY" -m sutra.cli.materialize_radius5 --project-root "$ROOT" --workers 3
"$PY" "$VERIFY" "$ROOT"
