from __future__ import annotations
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sutra.hierarchy.v094.diagnostics import load_landmark_table,analyze_landmarks

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def require(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p

def writepq(df,path):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),path,compression="zstd")

def one(project_s,sample,cfg):
    os.environ.update(
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1"
    )
    project=Path(project_s)
    src=project/"results"/"hierarchy_v093_full_completed_ultraslow_flow"
    scert=json.loads(require(src/sample/"full_completed_flow_certificate.json").read_text())
    n0=int(scert["level0_cells"])
    ldir=require(src/"ledger"/sample/"landmarks")

    df=load_landmark_table(ldir)
    diag,meta,windows,cps,recs,info=analyze_landmarks(df,n0,cfg)

    out=project/"results"/"hierarchy_v094_ness_flow_diagnostics"/sample
    out.mkdir(parents=True,exist_ok=True)
    writepq(diag,out/"landmark_flow_diagnostics.parquet")
    writepq(meta,out/"observable_normalization.parquet")
    writepq(windows,out/"stationary_windows.parquet")
    writepq(cps,out/"flow_change_points.parquet")
    writepq(recs,out/"candidate_effective_scales.parquet")

    report={
        "sample":sample,
        "level0_cells":n0,
        "terminal_nodes":int(scert["final_nodes"]),
        "terminal_removed_fraction":float(scert["removed_fraction"]),
        "landmarks_analyzed":int(len(df)),
        "observables_in_flow_metric":int(info["n_observables"]),
        "stationary_windows":int(info["n_stationary_windows"]),
        "candidate_effective_scales":recs.to_dict(orient="records"),
        "top_change_points":cps.head(8).to_dict(orient="records"),
        "policy":{
            "new_coarse_graining_performed":False,
            "v093_trajectory_modified":False,
            "terminal_2_3_node_state_allowed_as_biological_recommendation":False,
            "minimum_nontrivial_effective_nodes":int(cfg["min_effective_nodes"]),
            "scale_variable":"ell=log(N0/N)",
            "flow_metric":"RMS derivative of robust-standardized multivariate effective state",
            "acceleration_metric":"RMS second derivative of robust-standardized multivariate effective state",
            "single_scale_forced":False,
            "multiple_persistent_effective_regimes_allowed":True
        },
        "status":"PASS" if len(df)>=50 and info["n_observables"]>=8 else "HOLD"
    }
    (out/"ness_flow_diagnostics_certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v094_ness_flow_diagnostics.json")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    cfg=json.loads(require(project/a.config).read_text())

    c=require(project/"results"/"hierarchy_v093_full_completed_ultraslow_flow"/"full_completed_flow_global_certificate.json")
    d=json.loads(c.read_text())
    if d.get("FULL_MULTISCALE_TRAJECTORY_COMPLETE") is not True:
        raise SystemExit("ERROR: v0.9.3 exhaustive trajectory is not complete")

    print("STRATA 0.9.4 | NESS effective-state flow diagnostics")
    print("No new reduction is performed.")
    print("Mining the completed v0.9.3 trajectories from cells to exhaustive collapse.")
    print("Scale variable: ell = log(N0/N).")
    print("Searching for persistent low-flow regimes and multiscale change points.")
    print("Trivial 2-3 node terminal states cannot be recommended as biological scales.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        futs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(futs):
            r=f.result();reports.append(r)
            if r["candidate_effective_scales"]:
                top=r["candidate_effective_scales"][0]
                msg=f"top_scale_nodes={top['nodes']:,} removed={100*top['removed_fraction']:.1f}%"
            else:
                msg="top_scale=NONE"
            print(
                f"[DONE] {r['sample']}: landmarks={r['landmarks_analyzed']} "
                f"observables={r['observables_in_flow_metric']} "
                f"windows={r['stationary_windows']} {msg} {r['status']}",
                flush=True
            )

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v094_ness_flow_diagnostics"
    g={
        "strata_version":"0.9.4",
        "stage":"retrospective NESS effective-state flow diagnostics",
        "source_v093_trajectory_unchanged":True,
        "parallel_specimens":3,
        "sample_reports":reports,
        "NESS_FLOW_DIAGNOSTICS_GATE":"PASS" if gate else "HOLD",
        "CANDIDATE_EFFECTIVE_SCALES_READY":bool(gate),
        "READY_FOR_ADAPTIVE_STOPPING_RULE_DESIGN":bool(gate)
    }
    p=out/"ness_flow_diagnostics_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nNESS FLOW DIAGNOSTICS GATE: {g['NESS_FLOW_DIAGNOSTICS_GATE']}")
    print(f"CANDIDATE EFFECTIVE SCALES READY: {g['CANDIDATE_EFFECTIVE_SCALES_READY']}")
    print(f"READY FOR ADAPTIVE STOPPING RULE DESIGN: {g['READY_FOR_ADAPTIVE_STOPPING_RULE_DESIGN']}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
