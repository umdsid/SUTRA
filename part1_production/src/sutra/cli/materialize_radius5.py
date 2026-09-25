from __future__ import annotations
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from sutra.production_geometry.materialize_radius5 import materialize

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    cert=json.loads(
        (project/"results"/"tranche2_2e_radius_calibration"/"tranche2_2e_radius_freeze_certificate.json").read_text()
    )
    if cert.get("freeze_status")!="PASS" or float(cert.get("global_selected_radius"))!=5.0:
        raise SystemExit("Global radius 5.0 is not frozen.")

    samples=["alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"]
    reports=[]
    with ProcessPoolExecutor(max_workers=min(3,a.workers)) as ex:
        futs={ex.submit(materialize,project,s):s for s in samples}
        for fut in as_completed(futs):
            r=fut.result();reports.append(r)
            print(
                f"[DONE] {r['sample']}: cells={r['n_cells']:,} "
                f"LCC={100*r['largest_component_fraction']:.2f}% "
                f"recall={100*r['gateB_contact_recall']:.2f}% "
                f"new={100*r['new_contact_burden']:.2f}% "
                f"unassigned={100*r['remaining_unassigned_fraction']:.2f}%"
            )

    reports.sort(key=lambda x:samples.index(x["sample"]))
    ok=all(
        r["observed_pixels_preserved"] and
        r["gateB_contact_recall"]>=.95 and
        r["new_contact_burden"]<=.35 and
        r["largest_component_fraction"]>=.90
        for r in reports
    )
    out=project/"results"/"production_geometry_r5"
    c={
        "strata_version":"0.5.9",
        "production_method":"constrained_nearest",
        "frozen_radius":5.0,
        "sample_reports":reports,
        "gate_status":"PASS" if ok else "HOLD",
    }
    (out/"production_geometry_r5_certificate.json").write_text(json.dumps(c,indent=2))
    print()
    print(f"R5 PRODUCTION GEOMETRY: {c['gate_status']}")

if __name__=="__main__":main()
