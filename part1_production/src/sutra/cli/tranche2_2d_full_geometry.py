from __future__ import annotations
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed

from sutra.production_geometry.materialize import materialize_sample


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    freeze=project/"results"/"tranche2_2b_geometry_benchmark"/"tranche2_2c_freeze_certificate.json"
    if not freeze.exists():
        raise SystemExit("Missing Tranche 2.2c freeze certificate.")
    f=json.loads(freeze.read_text())
    if f.get("freeze_status")!="PASS" or f.get("production_geometry")!="constrained_nearest":
        raise SystemExit("Geometry is not frozen to constrained_nearest.")

    samples=["alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"]
    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        futs={ex.submit(materialize_sample,project,s):s for s in samples}
        for fut in as_completed(futs):
            s=futs[fut]
            r=fut.result()
            reports.append(r)
            print(
                f"[DONE] {s}: cells={r['n_cells']:,} "
                f"LCC={100*r['largest_component_fraction']:.2f}% "
                f"GateBrecall={100*r['gateB_contact_recall']:.2f}% "
                f"new={100*r['new_contact_burden']:.1f}%"
            )

    reports.sort(key=lambda x:samples.index(x["sample"]))
    # Production full-geometry gate: preserve observed cores; keep topology largely intact.
    gate_ok=all(
        r["observed_pixels_preserved"] and
        r["gateB_contact_recall"]>=0.95 and
        r["new_contact_burden"]<=0.35 and
        r["largest_component_fraction"]>=0.90
        for r in reports
    )
    cert={
        "strata_version":"0.5.7",
        "tranche":"2.2d",
        "production_geometry":"constrained_nearest",
        "scope":"full primary tissue component, all three specimens",
        "sample_reports":reports,
        "gate_status":"PASS" if gate_ok else "HOLD",
        "gate_rule":{
            "observed_pixels_preserved":True,
            "min_gateB_contact_recall":0.95,
            "max_new_contact_burden":0.35,
            "min_largest_component_fraction":0.90,
        },
    }
    out=project/"results"/"production_geometry"
    (out/"tranche2_2d_full_geometry_certificate.json").write_text(json.dumps(cert,indent=2))
    print()
    print(f"FULL GEOMETRY GATE: {cert['gate_status']}")
    print(f"Certificate: {out/'tranche2_2d_full_geometry_certificate.json'}")

if __name__=="__main__":main()
