#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
echo "[1/7] Python syntax"
"$PY" -m compileall -q "$HERE/part1_production/sutra" "$HERE/part1_production/tests" "$HERE/part2_paper/code" "$HERE/part2_paper/validate_frozen_products.py"
echo "[2/7] Shell syntax"
while IFS= read -r -d '' f; do bash -n "$f"; done < <(find "$HERE" -type f -name '*.sh' -print0)
echo "[3/7] Frozen paper products"
"$PY" "$HERE/part2_paper/validate_frozen_products.py"
echo "[4/7] Private-path scan"
! grep -RInE '/Users/sid|/home/sid' "$HERE/part1_production" "$HERE/part2_paper/code"
echo "[5/7] Historical-version exposure guard"
! find "$HERE/part1_production/sutra" -type d -regex '.*\/v0[0-9]+' -print -quit | grep -q .
! find "$HERE/part1_production" -maxdepth 1 -type f -regex '.*v0[0-9]+.*' -print -quit | grep -q .
echo "[6/7] Canonical entry point"
test -f "$HERE/part1_production/run_sutra.py"
test -d "$HERE/part1_production/sutra/hierarchy"
echo "[7/7] Full-figure guard"
! find "$HERE/part2_paper/main" -type f \( -iname '*composite*.png' -o -iname '*composite*.pdf' -o -iname '*fullfigure*.png' -o -iname '*fullfigure*.pdf' \) -print -quit | grep -q .
echo "ALL RC2 RELEASE TESTS PASSED"
