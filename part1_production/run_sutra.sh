#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
cat <<'EOF'
SUTRA — Spatial Unified Transcriptomic Reconstruction and Analysis

Usage:
  ./run_sutra.sh --help
  ./run_sutra.sh --preflight
  ./run_sutra.sh --run

--preflight  Verify local package ownership and required data/resources without running analysis.
--run        Execute the frozen five-specimen production pipeline.
EOF
}

case "${1:---help}" in
  --help|-h) usage ;;
  --preflight)
    export PYTHONPATH="$HERE/src"
    python - "$HERE" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
src=(root/"src").resolve()
import sutra
import sutra.hierarchy
for mod in (sutra, sutra.hierarchy):
    p=Path(mod.__file__).resolve()
    if src not in p.parents:
        raise SystemExit(f"ERROR: import outside SUTRA source: {p}")
print("SUTRA package ownership: PASS")
missing=[]
for name in ("data","resources"):
    p=root/name
    if not p.is_dir() or p.is_symlink(): missing.append(name)
if missing:
    print("Source installation: PASS")
    print("Production inputs not installed:", ", ".join(missing))
    print("Populate data/ and resources/ before --run.")
else:
    print("Production inputs: PASS")
PY
    ;;
  --run)
    exec bash "$HERE/run_sutra_five_v0911_full.sh" "$HERE"
    ;;
  *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
esac
