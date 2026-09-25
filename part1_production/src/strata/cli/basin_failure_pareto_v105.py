from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

from strata_hierarchy.v105.pareto_audit import (
    audit_v104_binary_failures,criterion_summary,coherence_distribution,
    threshold_sensitivity,pareto_layers,nearest_scale_spacing,
    attach_state_descriptors,block_failure_profile,
    pareto_stability_leave_one_metric_out
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def req(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p

def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def one(project_s,sample,cfg):
    project=Path(project_s)
    src=project/"results"/"hierarchy_v104_persistence_basin_landmarks"/sample

    landmarks=pd.read_parquet(req(src/"hierarchy_landmarks.parquet"))
    state=pd.read_parquet(req(src/"landmark_spatial_gene_state.parquet"))
    block_detail=pd.read_parquet(req(src/"basin_block_detail.parquet"))

    audited=audit_v104_binary_failures(landmarks,cfg)
    audited=nearest_scale_spacing(audited)
    audited=attach_state_descriptors(audited,state)

    crit=criterion_summary(audited)
    coh=coherence_distribution(audited)
    sens=threshold_sensitivity(audited,cfg["coherence_threshold_audit_grid"])
    bprof=block_failure_profile(block_detail)

    metrics=["lifetime_ell","basin_depth","cross_block_coherence"]
    layered=pareto_layers(audited,metrics,int(cfg["max_pareto_layers"]))
    loo=pareto_stability_leave_one_metric_out(audited,metrics)

    if len(bprof) and "basin_id" in layered.columns:
        layered=layered.merge(bprof,on="basin_id",how="left",suffixes=("","_blockaudit"))

    front=layered[layered.pareto_layer==1].copy()
    # A Pareto basin is an audit survivor, not automatically a production landmark.
    front["pareto_audit_survivor"]=True

    out=project/"results"/"hierarchy_v105_basin_failure_pareto_audit"/sample
    out.mkdir(parents=True,exist_ok=True)
    wpq(audited,out/"v104_binary_failure_audit.parquet")
    wpq(crit,out/"binary_criterion_summary.parquet")
    wpq(sens,out/"coherence_threshold_sensitivity.parquet")
    wpq(layered,out/"pareto_hierarchy_layers.parquet")
    wpq(front,out/"pareto_front_basins.parquet")
    wpq(loo,out/"pareto_leave_one_metric_out.parquet")
    wpq(bprof,out/"basin_block_failure_profile.parquet")

    all_reproduced=bool(audited.v104_flag_agrees.all()) if len(audited) else False
    failure_counts=(
        audited.binary_failure_reasons.value_counts(dropna=False).to_dict()
        if len(audited) else {}
    )
    report={
        "sample":sample,
        "basins":int(len(audited)),
        "v104_binary_landmarks":int(audited.hierarchy_landmark.astype(bool).sum()),
        "v104_binary_decisions_reproduced":all_reproduced,
        "failure_reason_counts":{str(k):int(v) for k,v in failure_counts.items()},
        "coherence_distribution":coh,
        "pareto_front_size":int(len(front)),
        "pareto_front_nodes":[int(x) for x in front.minimum_nodes.tolist()],
        "pareto_front_removed_fractions":[float(x) for x in front.minimum_removed_fraction.tolist()],
        "pareto_metrics":metrics,
        "entropy_used_as_optimization_objective":False,
        "spatial_or_gene_descriptor_used_as_gate":False,
        "coherence_threshold_changed":False,
        "threshold_sensitivity_is_diagnostic_only":True,
        "new_coarse_graining_performed":False,
        "status":"PASS" if all_reproduced and len(front)>=1 else "HOLD"
    }
    (out/"basin_failure_pareto_certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v105_basin_failure_pareto.json")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())

    cert=json.loads(req(
        project/"results"/"hierarchy_v104_persistence_basin_landmarks"/
        "persistence_basin_landmark_global_certificate.json"
    ).read_text())
    if cert.get("strata_version")!="1.0.4":
        raise SystemExit("ERROR: v1.0.4 persistence-basin certificate missing")

    print("STRATA 1.0.5 | Basin-failure + Pareto-hierarchy audit")
    print("No new coarse-graining and no v1.0.4 threshold changes.")
    print("Every basin receives an explicit binary failure explanation.")
    print("Coherence-threshold sensitivity is diagnostic only; no new cutoff is selected.")
    print("Primary Pareto objectives: lifetime, depth, cross-block coherence.")
    print("Switching entropy is descriptive, not automatically rewarded or penalized.")
    print("Spatial/gene descriptors are attached when present but do not gate the front.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result(); reps.append(r)
            c=r["coherence_distribution"]
            print(
                f"[DONE] {r['sample']}: basins={r['basins']} "
                f"binary={r['v104_binary_landmarks']} "
                f"coh_med={c['median']:.3f} coh_max={c['max']:.3f} "
                f"pareto={r['pareto_front_size']} nodes={r['pareto_front_nodes']} "
                f"{r['status']}",flush=True
            )

    reps.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reps)
    out=project/"results"/"hierarchy_v105_basin_failure_pareto_audit"
    out.mkdir(parents=True,exist_ok=True)
    g={
        "strata_version":"1.0.5",
        "stage":"basin failure and Pareto hierarchy audit",
        "sample_reports":reps,
        "BASIN_FAILURE_AUDIT_GATE":"PASS" if gate else "HOLD",
        "V104_BINARY_CLASSIFIER_REPRODUCED":bool(
            all(r["v104_binary_decisions_reproduced"] for r in reps)
        ),
        "PARETO_BASIN_LANDSCAPE_READY":bool(gate),
        "PRODUCTION_LANDMARK_RULE_FROZEN":False,
        "NEW_COARSE_GRAINING_PERFORMED":False,
        "READY_TO_FREEZE_LANDMARK_DEFINITION":bool(gate)
    }
    p=out/"basin_failure_pareto_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")

    print(f"\nBASIN FAILURE AUDIT GATE: {g['BASIN_FAILURE_AUDIT_GATE']}")
    print(f"V1.0.4 BINARY CLASSIFIER REPRODUCED: {g['V104_BINARY_CLASSIFIER_REPRODUCED']}")
    print(f"PARETO BASIN LANDSCAPE READY: {g['PARETO_BASIN_LANDSCAPE_READY']}")
    print("PRODUCTION LANDMARK RULE FROZEN: False")
    print(f"READY TO FREEZE LANDMARK DEFINITION: {g['READY_TO_FREEZE_LANDMARK_DEFINITION']}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
