#!/usr/bin/env python3
from pathlib import Path
import json
import sys

root = Path(sys.argv[1]).resolve()
cert = root / "results" / "production_geometry_r5" / "production_geometry_r5_certificate.json"

if not cert.is_file():
    raise SystemExit(2)

j = json.loads(cert.read_text())
if j.get("gate_status") != "PASS":
    raise SystemExit(f"ERROR: radius-5 production geometry gate is {j.get('gate_status')}")

samples = [
    "alzheimers",
    "gbm_reference_addon",
    "healthy_reference",
    "nondiseased_kidney",
    "prcc",
]

for s in samples:
    q = root / "results" / "production_geometry_r5" / s / "production_segmentation.tif"
    if not q.is_file() or q.stat().st_size == 0:
        raise SystemExit(f"ERROR: missing production segmentation: {q}")

print("RADIUS-5 PRODUCTION GEOMETRY: PASS")
for r in j.get("sample_reports", []):
    print(
        f"{r['sample']:24s} "
        f"cells={int(r['n_cells']):8,d} "
        f"LCC={100*float(r['largest_component_fraction']):6.2f}% "
        f"recall={100*float(r['gateB_contact_recall']):6.2f}% "
        f"new={100*float(r['new_contact_burden']):6.2f}%"
    )
