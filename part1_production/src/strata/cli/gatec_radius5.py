from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

TOKENS=("pressure","tension","stress")

def audit(project,sample):
    geom=json.loads(
        (project/"results"/"production_geometry_r5"/sample/"production_geometry_summary.json").read_text()
    )
    out=project/"results"/"production_mechanics_vmsi_r5"/sample
    p=out/"vmsi_results.csv"
    if not p.exists():
        return {"sample":sample,"status":"HOLD","reason":"VMSI output missing"}

    df=pd.read_csv(p)
    num=df.select_dtypes(include=[np.number])
    mech=[c for c in num.columns if any(t in c.lower() for t in TOKENS)]
    finite={}
    std={}
    for c in mech:
        x=pd.to_numeric(df[c],errors="coerce").to_numpy(float)
        q=np.isfinite(x)
        finite[c]=float(q.mean())
        std[c]=float(np.nanstd(x)) if q.any() else None

    coverage=min(len(df)/max(int(geom["n_cells"]),1),1.0)
    finite_min=min(finite.values()) if finite else 0.0
    nontrivial=sum(1 for v in std.values() if v is not None and np.isfinite(v) and v>0)

    ok=(coverage>=.85 and len(mech)>=1 and finite_min>=.95 and nontrivial>=1)
    return {
        "sample":sample,
        "status":"PASS" if ok else "HOLD",
        "cell_result_coverage":float(coverage),
        "mechanical_columns":mech,
        "finite_fraction_by_column":finite,
        "std_by_column":std,
        "nontrivial_mechanical_columns":nontrivial,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    samples=["alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"]
    reports=[audit(project,s) for s in samples]
    overall={
        "strata_version":"0.5.9",
        "gate":"C-production-VMSI-radius5",
        "sample_reports":reports,
        "gate_status":"PASS" if all(r["status"]=="PASS" for r in reports) else "HOLD",
    }
    out=project/"results"/"production_mechanics_vmsi_r5"
    out.mkdir(parents=True,exist_ok=True)
    (out/"gateC_radius5_certificate.json").write_text(json.dumps(overall,indent=2))
    for r in reports:
        print(f"[{r['sample']}] {r['status']} coverage={100*r.get('cell_result_coverage',0):.1f}% fields={len(r.get('mechanical_columns',[]))}")
    print()
    print(f"GATE C: {overall['gate_status']}")

if __name__=="__main__":main()
