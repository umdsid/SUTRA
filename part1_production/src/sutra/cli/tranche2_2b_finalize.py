from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np


def metric_value(m,key,default=None):
    try:return m["score"][key]
    except Exception:return default


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    root=project/"results"/"tranche2_2b_geometry_benchmark"
    samples=["alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"]
    reports=[]
    for s in samples:
        out=root/s
        prep=json.loads((out/"prepare_summary.json").read_text())
        methods=prep["methods"]
        ps=out/"proseg_score.json"
        if ps.exists():
            methods["proseg"]={"score":json.loads(ps.read_text())}

        # Conservative selection score. Connectivity is intentionally NOT included.
        scored=[]
        for name,m in methods.items():
            sc=m.get("score")
            if not sc: continue
            f1=float(sc.get("contact_f1_vs_gateB",0))
            cap=float(sc.get("transcript_capture_fraction",
                             sc.get("transcript_assignment",{}).get("assigned_fraction",0) or 0))
            new=float(sc.get("new_contact_burden",1))
            distort=float(sc.get("median_abs_log_area_ratio",0))
            # Weight evidence preservation over graph completion.
            total=0.45*f1+0.35*cap+0.20*(1-new)-0.05*distort
            scored.append((total,name))
        scored.sort(reverse=True)

        winner=scored[0][1] if scored else None
        report={
            "sample":s,
            "primary_component":prep["primary_component"],
            "methods":methods,
            "ranking":[{"method":n,"selection_score":float(v)} for v,n in scored],
            "provisional_winner":winner,
            "selection_status":"PASS" if winner else "HOLD",
            "note":"Connectivity is diagnostic only and is not part of the selection objective."
        }
        (out/"geometry_benchmark_certificate.json").write_text(json.dumps(report,indent=2))
        reports.append(report)

    overall={
        "strata_version":"0.5.1",
        "tranche":"2.2b",
        "sample_reports":reports,
        "status":"PASS" if all(r["selection_status"]=="PASS" for r in reports) else "HOLD",
        "production_rule":"No reconstructed geometry replaces Level-0 observed segmentation. A selected mechanics geometry must remain separately versioned and provenance-tracked."
    }
    (root/"tranche2_2b_certificate.json").write_text(json.dumps(overall,indent=2))
    print(json.dumps({"status":overall["status"],
                      "winners":{r["sample"]:r["provisional_winner"] for r in reports}},indent=2))

if __name__=="__main__":main()
