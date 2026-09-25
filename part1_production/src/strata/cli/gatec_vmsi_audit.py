from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd


MECH_TOKENS=("pressure","tension","stress")


def audit_one(project,sample):
    geom=json.loads(
        (project/"results"/"production_geometry"/sample/"production_geometry_summary.json").read_text()
    )
    vdir=project/"results"/"production_mechanics_vmsi"/sample
    csv=vdir/"vmsi_results.csv"
    status_file=vdir/"vmsi_run_status.json"
    if not csv.exists():
        return {
            "sample":sample,"status":"FAIL",
            "reason":"VMSI result table missing",
            "vmsi_status":json.loads(status_file.read_text()) if status_file.exists() else None,
        }

    df=pd.read_csv(csv)
    numeric=df.select_dtypes(include=[np.number])
    mech=[c for c in numeric.columns if any(tok in c.lower() for tok in MECH_TOKENS)]
    finite={}
    variable={}
    for c in mech:
        x=pd.to_numeric(df[c],errors="coerce").to_numpy(float)
        q=np.isfinite(x)
        finite[c]=float(q.mean())
        variable[c]=float(np.nanstd(x)) if q.any() else None

    n_cells=int(geom["n_cells"])
    coverage=min(len(df)/max(n_cells,1),1.0)
    finite_min=min(finite.values()) if finite else 0.0
    nontrivial=sum(
        1 for v in variable.values()
        if v is not None and np.isfinite(v) and v>0
    )

    # Gate C is now a mechanics-readiness/execution gate on the corrected geometry.
    # It does not pretend that finite output alone proves biological identifiability.
    passed=(
        coverage>=0.85 and
        len(mech)>=1 and
        finite_min>=0.95 and
        nontrivial>=1
    )
    return {
        "sample":sample,
        "status":"PASS" if passed else "HOLD",
        "n_geometry_cells":n_cells,
        "n_vmsi_rows":len(df),
        "cell_result_coverage":float(coverage),
        "mechanical_columns":mech,
        "finite_fraction_by_column":finite,
        "std_by_column":variable,
        "nontrivial_mechanical_columns":nontrivial,
        "note":"Gate C certifies successful mechanics on the corrected geometry; identifiability remains a separate audit."
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    samples=["alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"]
    reports=[audit_one(project,s) for s in samples]
    overall={
        "strata_version":"0.5.7",
        "gate":"C-production-VMSI",
        "sample_reports":reports,
        "gate_status":"PASS" if all(r["status"]=="PASS" for r in reports) else "HOLD",
        "interpretation":"mechanics execution/coverage gate on the frozen completed geometry",
    }
    out=project/"results"/"production_mechanics_vmsi"
    out.mkdir(parents=True,exist_ok=True)
    (out/"gateC_production_certificate.json").write_text(json.dumps(overall,indent=2))
    for r in reports:
        print(
            f"[{r['sample']}] {r['status']} "
            f"coverage={100*r.get('cell_result_coverage',0):.1f}% "
            f"mechanical_fields={len(r.get('mechanical_columns',[]))}"
        )
    print()
    print(f"GATE C: {overall['gate_status']}")
    print(f"Certificate: {out/'gateC_production_certificate.json'}")

if __name__=="__main__":main()
