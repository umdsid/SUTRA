#!/usr/bin/env python3
from pathlib import Path
from PIL import Image
import hashlib, sys
ROOT=Path(__file__).resolve().parent
errors=[]
def check_png(p):
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            if im.width < 100 or im.height < 100: errors.append(f"too small: {p}")
    except Exception as e: errors.append(f"bad PNG {p}: {e}")
for p in ROOT.rglob("*.png"): check_png(p)
for p in ROOT.rglob("*.pdf"):
    if p.read_bytes()[:4] != b"%PDF": errors.append(f"bad PDF: {p}")
# No assembled main figures are allowed.
for bad in ("Figure2","Figure3","Fig2_full","Fig3_full","composite"):
    for p in ROOT.rglob(f"*{bad}*"):
        if p.is_file() and p.suffix.lower() in (".png",".pdf"):
            errors.append(f"assembled/full figure forbidden: {p}")
expected=[ROOT/f"main/fig2/panels/Fig2{x}.png" for x in "ABCDEFGH"]
expected += [ROOT/f"main/fig3/panels/{x}.png" for x in
 ["Fig3A","Fig3B","Fig3C","Fig3D","Fig3E_brain","Fig3E_kidney","Fig3F_brain","Fig3F_kidney"]]
for p in expected:
    if not p.exists(): errors.append(f"missing approved panel: {p}")
if errors:
    print("\n".join("ERROR: "+e for e in errors)); sys.exit(1)
print(f"PASS: {len(list(ROOT.rglob('*.png')))} PNGs and {len(list(ROOT.rglob('*.pdf')))} PDFs validated.")
print("PASS: approved Fig2/Fig3 panel inventory present; no assembled main figures shipped.")
