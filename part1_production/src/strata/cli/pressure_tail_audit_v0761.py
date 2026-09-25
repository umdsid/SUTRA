from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from strata_hierarchy.v071.resource_intake import sha256_file
from strata_hierarchy.v076.pressure_tail_audit import (
    reconstruct_pressure_residuals,robust_tail_threshold,attach_tail_flags,
    pressure_components,level1_selection_enrichment,summarize_tail
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def require(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p

def writepq(df,path):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),path,compression="zstd")

def one(project,sample):
    l0=project/"results"/"hierarchy_level0_v070"/sample
    v72=project/"results"/"hierarchy_v072_short_pilot"/sample
    v76=project/"results"/"hierarchy_v076_geometry_aware"/sample

    cells=pd.read_parquet(require(l0/"cells.parquet"))
    edge_rel=pd.read_parquet(require(v72/"level0_candidate_relations.parquet"))

    p,E,solver=reconstruct_pressure_residuals(edge_rel,len(cells))
    tail=robust_tail_threshold(E.abs_pressure_residual.to_numpy(np.float64))
    E=attach_tail_flags(E,tail)
    labels,components=pressure_components(E,len(cells))

    bounds=pd.read_parquet(require(v76/"boundary_geometry_all_levels.parquet"))
    level1=bounds[bounds.level==1].copy() if len(bounds) else pd.DataFrame()
    enrichment=level1_selection_enrichment(E,level1)
    summary=summarize_tail(E,tail,components,enrichment)

    old=json.loads(require(v76/"geometry_aware_hierarchy_summary.json").read_text())["pressure_reconstruction"]
    reproduce={
        "median_abs_error":float(abs(summary["median_abs"]-float(old["residual_median_abs"]))),
        "q95_abs_error":float(abs(summary["q95_abs"]-float(old["residual_q95_abs"]))),
        "rms_abs_error":float(abs(summary["rms"]-float(old["residual_rms"]))),
    }
    reproduce["matches_v076"]=bool(
        reproduce["median_abs_error"]<=1e-8 and
        reproduce["q95_abs_error"]<=1e-8 and
        reproduce["rms_abs_error"]<=max(1e-6,1e-10*abs(float(old["residual_rms"])))
    )

    out=project/"results"/"hierarchy_v0761_pressure_tail_audit"/sample
    out.mkdir(parents=True,exist_ok=True)
    writepq(E,out/"pressure_residual_edge_audit.parquet")
    writepq(components,out/"pressure_component_audit.parquet")
    writepq(E.nlargest(min(500,len(E)),"abs_pressure_residual"),
            out/"largest_pressure_residuals.parquet")

    report={
        "sample":sample,
        "solver":solver,
        "tail_summary":summary,
        "v076_reproduction":reproduce,
        "hierarchy_modified":False,
        "geometry_modified":False,
        "status":"PASS" if summary["all_residuals_finite"] and reproduce["matches_v076"] else "HOLD",
    }
    (out/"pressure_tail_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    src=require(project/"results"/"hierarchy_v076_geometry_aware"/"geometry_aware_hierarchy_certificate.json")
    d=json.loads(src.read_text())
    if d.get("GEOMETRY_DRIVES_HIERARCHY") is not True:
        raise SystemExit("ERROR: v0.7.6 geometry-aware hierarchy did not pass")

    print("STRATA 0.7.6.1 | Pressure residual tail audit")
    print("Diagnostic only: no hierarchy or geometry is modified.")
    print("Auditing extreme least-squares residuals and Level-1 merge enrichment.\n")

    reports=[]
    for s in SAMPLES:
        r=one(project,s); reports.append(r)
        q=r["tail_summary"]
        e=q["level1_selection_enrichment"]
        enrich=e["extreme_enrichment_ratio"]
        estr="NA" if not np.isfinite(enrich) else f"{enrich:.3g}x"
        print(
            f"[DONE] {s}: median={q['median_abs']:.3e} "
            f"q99={q['q99_abs']:.3e} q99.9={q['q999_abs']:.3e} "
            f"max={q['max_abs']:.3e} extreme={q['n_extreme_tail']}/{q['n_constraints']} "
            f"components={q['n_components_with_extreme_tail']}/{q['n_components']} "
            f"L1_enrich={estr} {r['status']}"
        )

    gate=all(r["status"]=="PASS" for r in reports)
    sparse_tail_all=all(r["tail_summary"]["sparse_tail_dominates_rms"] for r in reports)

    out=project/"results"/"hierarchy_v0761_pressure_tail_audit"
    cert={
        "strata_version":"0.7.6.1",
        "stage":"pressure residual tail audit",
        "source_v076_certificate_sha256":sha256_file(src),
        "policy":{
            "diagnostic_only":True,
            "hierarchy_changed":False,
            "geometry_changed":False,
            "pressure_constraints_removed":False,
            "outliers_clipped":False,
        },
        "sample_reports":reports,
        "PRESSURE_TAIL_AUDIT_GATE":"PASS" if gate else "HOLD",
        "SPARSE_TAIL_EXPLAINS_RMS_ALL_SAMPLES":bool(sparse_tail_all),
        "READY_TO_DECIDE_PRESSURE_TAIL_POLICY":bool(gate),
    }
    p=out/"pressure_tail_audit_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")
    print(f"\nPRESSURE TAIL AUDIT GATE: {cert['PRESSURE_TAIL_AUDIT_GATE']}")
    print(f"SPARSE TAIL EXPLAINS RMS ALL SAMPLES: {cert['SPARSE_TAIL_EXPLAINS_RMS_ALL_SAMPLES']}")
    print(f"READY TO DECIDE PRESSURE TAIL POLICY: {cert['READY_TO_DECIDE_PRESSURE_TAIL_POLICY']}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
