from __future__ import annotations
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sutra.hierarchy.v0941.audit import (
    compute_metric,contribution_table,audit_grid,baseline_stability,change_point_indices
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def req(p):
    p=Path(p)
    if not p.exists():raise FileNotFoundError(p)
    return p

def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def one(project_s,sample,cfg):
    os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                      VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1")
    project=Path(project_s)
    src=project/"results"/"hierarchy_v094_ness_flow_diagnostics"/sample
    df=pd.read_parquet(req(src/"landmark_flow_diagnostics.parquet"))

    # Recompute baseline with equal weight per scientific block.
    vel,acc,wins,bv,ba,colmap=compute_metric(
        df,cfg,cfg["baseline_smoothing_width"],cfg["baseline_derivative_stride"],balanced=True
    )
    out=project/"results"/"hierarchy_v0941_flow_metric_audit"/sample
    out.mkdir(parents=True,exist_ok=True)

    base=df[["landmark_index","microstep","removed_fraction","nodes","ell"]].copy()
    base["block_balanced_velocity"]=vel
    base["block_balanced_acceleration"]=acc
    wpq(base,out/"block_balanced_flow.parquet")

    # Contributions at all landmarks and focused contributions at original change points.
    contrib=contribution_table(df,vel,bv)
    wpq(contrib,out/"block_velocity_contributions.parquet")
    cpidx=change_point_indices(df,cfg["change_point_top_k"])
    cpcon=contribution_table(df,vel,bv,cpidx)
    wpq(cpcon,out/"change_point_block_contributions.parquet")

    grid,allwins=audit_grid(df,cfg)
    wpq(grid,out/"metric_parameter_sweep_windows.parquet")
    stability=baseline_stability(grid,allwins,cfg)
    wpq(stability,out/"baseline_window_stability.parquet")

    # Aggregate block shares within stable baseline windows.
    rows=[]
    for rank,w in enumerate(wins,1):
        q=(contrib.table_index>=w["start"])&(contrib.table_index<=w["end"])
        g=contrib[q].groupby("block",as_index=False).agg(
            mean_squared_velocity_fraction=("squared_velocity_fraction","mean"),
            mean_block_velocity=("block_velocity_rms","mean")
        )
        g.insert(0,"baseline_window_rank",rank)
        g["nodes_rep"]=w["nodes_rep"];g["removed_rep"]=w["removed_rep"]
        rows.append(g)
    wincon=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    wpq(wincon,out/"stationary_window_block_contributions.parquet")

    stable=stability[stability.stable] if len(stability) else stability
    top=None
    if len(stable):
        z=stable.sort_values(
            ["support_fraction","mean_best_window_overlap","median_rep_ell_drift"],
            ascending=[False,False,True]
        ).iloc[0]
        top={k:(bool(v) if isinstance(v,(bool,)) else float(v) if hasattr(v,"item") else v)
             for k,v in z.to_dict().items()}
        top["nodes"]=int(z["nodes"])

    report={
        "sample":sample,
        "n_landmarks":int(len(df)),
        "n_observables":int(len(colmap)),
        "n_blocks":int(len(set(colmap.values()))),
        "baseline_block_balanced_windows":int(len(wins)),
        "stable_baseline_windows":int(len(stable)),
        "top_robust_scale":top,
        "parameter_grid":{
            "smoothing_widths":cfg["smoothing_widths"],
            "derivative_strides":cfg["derivative_strides"],
            "weighting_modes":["observable_balanced","block_balanced"],
            "total_configurations":2*len(cfg["smoothing_widths"])*len(cfg["derivative_strides"])
        },
        "policy":{
            "new_reduction_performed":False,
            "original_v094_metric_overwritten":False,
            "equal_weight_per_scientific_block_tested":True,
            "metric_resolution_sweep_tested":True,
            "block_contribution_decomposition_exported":True,
        },
        "status":"PASS" if len(stable)>0 and len(set(colmap.values()))>=6 else "HOLD"
    }
    (out/"flow_metric_audit_certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v0941_flow_metric_audit.json")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())

    gc=json.loads(req(
        project/"results"/"hierarchy_v094_ness_flow_diagnostics"/
        "ness_flow_diagnostics_global_certificate.json"
    ).read_text())
    if gc.get("CANDIDATE_EFFECTIVE_SCALES_READY") is not True:
        raise SystemExit("ERROR: v0.9.4 candidates not certified")

    print("STRATA 0.9.4.1 | Flow-metric robustness audit")
    print("No new coarse-graining.")
    print("Auditing modality dominance, equal-block weighting, and resolution stability.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result();reports.append(r)
            t=r["top_robust_scale"]
            msg=(f"robust_nodes={t['nodes']:,} support={t['support_fraction']:.2f}"
                 if t else "robust_scale=NONE")
            print(f"[DONE] {r['sample']}: blocks={r['n_blocks']} "
                  f"stable_windows={r['stable_baseline_windows']} {msg} {r['status']}",flush=True)

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v0941_flow_metric_audit"
    g={
        "strata_version":"0.9.4.1",
        "stage":"NESS flow metric robustness audit",
        "sample_reports":reports,
        "FLOW_METRIC_AUDIT_GATE":"PASS" if gate else "HOLD",
        "ROBUST_EFFECTIVE_REGIMES_READY":bool(gate),
        "READY_FOR_ADAPTIVE_NESS_FLOW_DESIGN":bool(gate)
    }
    p=out/"flow_metric_audit_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nFLOW METRIC AUDIT GATE: {g['FLOW_METRIC_AUDIT_GATE']}")
    print(f"ROBUST EFFECTIVE REGIMES READY: {g['ROBUST_EFFECTIVE_REGIMES_READY']}")
    print(f"READY FOR ADAPTIVE NESS FLOW DESIGN: {g['READY_FOR_ADAPTIVE_NESS_FLOW_DESIGN']}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
