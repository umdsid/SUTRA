#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
echo "[1/5] Python syntax"
"$PY" -m compileall -q "$HERE/part1_production/src" "$HERE/part1_production/tests" "$HERE/part2_paper/code" "$HERE/part2_paper/validate_frozen_products.py"
echo "[2/5] Shell syntax"
while IFS= read -r -d '' f; do bash -n "$f"; done < <(find "$HERE" -type f -name '*.sh' -print0)
echo "[3/5] Frozen product validation"
"$PY" "$HERE/part2_paper/validate_frozen_products.py"
echo "[4/5] Private absolute-path scan"
if grep -RInE '/Users/sid|/home/sid' "$HERE/part1_production/src" "$HERE/part1_production/configs" "$HERE/part1_production/tests" "$HERE/part2_paper/code"; then
  echo "ERROR: private absolute path found"; exit 1
fi
echo "[5/5] Full-figure output guard"
find "$HERE/part2_paper/main" -type f \( -iname '*composite*.png' -o -iname '*composite*.pdf' -o -iname '*fullfigure*.png' -o -iname '*fullfigure*.pdf' \) -print -quit | grep -q . && { echo "ERROR: assembled figure found"; exit 1; } || true
echo "ALL RELEASE TESTS PASSED"
