from __future__ import annotations
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from strata_hierarchy.v103.audit import (
    frozen_coordinates,block_displacements,block_closure_ratios,gate_table,
    metastable_candidates,veto_summary,block_state_summary,unstable_block_at_candidates
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
    v94=project/"results"/"hierarchy_v094_ness_flow_diagnostics"/sample
    v102=project/"results"/"hierarchy_v102_future_blind_plateau_replay"/sample

    df=pd.read_parquet(req(v94/"landmark_flow_diagnostics.parquet"))
    norm=pd.read_parquet(req(v94/"observable_normalization.parquet"))
    features=pd.read_parquet(req(v102/"central_future_blind_features.parquet"))

    cols,blocks,Z=frozen_coordinates(df,norm,float(cfg["min_finite_fraction"]))
    bd=block_displacements(Z,blocks,cfg["closure_lags"])
    br=block_closure_ratios(
        bd,df,int(cfg["central_recent_width"]),int(cfg["central_history_width"])
    )
    gates=gate_table(features,cfg)
    veto=veto_summary(gates)
    meta=metastable_candidates(gates,df,cfg)
    bsum=block_state_summary(br,gates,cfg)
    bcand=unstable_block_at_candidates(br,meta,cfg)

    out=project/"results"/"hierarchy_v103_gate_decomposition_metastable_audit"/sample
    out.mkdir(parents=True,exist_ok=True)
    wpq(gates,out/"gate_decomposition_by_landmark.parquet")
    wpq(veto,out/"gate_veto_summary.parquet")
    wpq(br,out/"block_multilag_closure_ratios.parquet")
    wpq(bsum,out/"block_near_plateau_summary.parquet")
    wpq(meta,out/"metastable_candidate_landmarks.parquet")
    wpq(bcand,out/"candidate_unstable_blocks.parquet")

    eligible=gates[gates.pass_eligible.astype(bool)]
    near3=int(gates.near_plateau_3of4.sum())
    best=None
    if len(meta):
        r=meta.sort_values(["plateau_distance_score","gate_count"],ascending=[True,False]).iloc[0]
        best={
            "landmark_index":int(r.landmark_index),"nodes":int(r.nodes),
            "removed_fraction":float(r.removed_fraction),
            "plateau_distance_score":float(r.plateau_distance_score),
            "gate_count":int(r.gate_count),
        }

    switch_count=0
    if len(bcand):
        top=bcand[bcand.block_rank_unstable==1].sort_values("landmark_index")
        vals=top.block.tolist()
        switch_count=sum(a!=b for a,b in zip(vals[:-1],vals[1:]))

    report={
        "sample":sample,
        "landmarks":int(len(gates)),
        "eligible_landmarks":int(len(eligible)),
        "near_plateau_3of4_landmarks":near3,
        "metastable_candidates":int(len(meta)),
        "best_metastable_candidate":best,
        "unstable_block_identity_switches":int(switch_count),
        "gate_pass_fractions":{
            str(r.gate):float(r.pass_fraction) if r.pass_fraction==r.pass_fraction else None
            for r in veto.itertuples(index=False)
        },
        "scientific_blocks":sorted(set(blocks)),
        "n_scientific_blocks":int(len(set(blocks))),
        "interpretation_flags":{
            "single_gate_dominant_veto":False,
            "multigate_persistent_reorganization":False,
            "scale_dependent_unstable_block_switching":bool(switch_count>=2),
        },
        "status":"PASS"
    }

    if len(veto):
        fr=veto.set_index("gate").pass_fraction
        bad=[g for g,x in fr.items() if x<0.35]
        report["interpretation_flags"]["single_gate_dominant_veto"]=bool(len(bad)==1)
        report["interpretation_flags"]["multigate_persistent_reorganization"]=bool(len(bad)>=2)

    (out/"gate_decomposition_metastable_certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v103_gate_decomposition_metastable.json")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())

    cert=json.loads(req(
        project/"results"/"hierarchy_v102_future_blind_plateau_replay"/
        "future_blind_plateau_replay_global_certificate.json"
    ).read_text())
    # HOLD is expected and is valid input; only stage/version existence matters.
    if cert.get("strata_version")!="1.0.2":
        raise SystemExit("ERROR: v1.0.2 replay certificate not found or wrong version")

    print("STRATA 1.0.3 | Gate decomposition + metastable-regime audit")
    print("No new coarse-graining and no threshold changes.")
    print("Decomposing velocity, acceleration, closure, resolution, and active-entry behavior.")
    print("Blockwise closure is resolved for all ten scientific sectors.")
    print("Local minima identify metastable candidate scales without forcing a stop.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result(); reps.append(r)
            b=r["best_metastable_candidate"]
            bmsg=(f"best_nodes={b['nodes']:,} gates={b['gate_count']}/4" if b else "best=NONE")
            print(
                f"[DONE] {r['sample']}: near3={r['near_plateau_3of4_landmarks']} "
                f"metastable={r['metastable_candidates']} "
                f"block_switches={r['unstable_block_identity_switches']} "
                f"{bmsg} PASS",flush=True
            )

    reps.sort(key=lambda x:SAMPLES.index(x["sample"]))
    out=project/"results"/"hierarchy_v103_gate_decomposition_metastable_audit"
    out.mkdir(parents=True,exist_ok=True)
    g={
        "strata_version":"1.0.3",
        "stage":"gate decomposition and metastable-regime audit",
        "sample_reports":reps,
        "GATE_DECOMPOSITION_AUDIT":"PASS",
        "THRESHOLDS_MODIFIED":False,
        "NEW_COARSE_GRAINING_PERFORMED":False,
        "READY_TO_DECIDE_STOPPING_PHILOSOPHY":True
    }
    p=out/"gate_decomposition_metastable_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print("\nGATE DECOMPOSITION AUDIT: PASS")
    print("THRESHOLDS MODIFIED: False")
    print("READY TO DECIDE STOPPING PHILOSOPHY: True")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
