from __future__ import annotations
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from sutra.hierarchy.v104.basins import (
    detect_basins,attach_block_coherence,hierarchy_landmarks,
    find_landmark_node_state,spatial_stats_from_node_state,gene_state_stats_from_node_state
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
    v103=project/"results"/"hierarchy_v103_gate_decomposition_metastable_audit"/sample
    gates=pd.read_parquet(req(v103/"gate_decomposition_by_landmark.parquet"))
    candidates=pd.read_parquet(req(v103/"metastable_candidate_landmarks.parquet"))
    block_ratios=pd.read_parquet(req(v103/"block_multilag_closure_ratios.parquet"))

    basins=detect_basins(gates,candidates,cfg)
    coh,detail=attach_block_coherence(basins,block_ratios,cfg)
    landmarks=hierarchy_landmarks(basins,coh,cfg)

    ledger_root=project/"results"/"hierarchy_v093_full_completed_ultraslow_flow"/"ledger"
    spatial_rows=[]; ancestry_available=0
    if len(landmarks):
        for r in landmarks.itertuples(index=False):
            state,path=find_landmark_node_state(ledger_root,sample,int(r.candidate_landmark))
            rec={"candidate_landmark":int(r.candidate_landmark),
                 "node_state_available":bool(state is not None),
                 "node_state_path":str(path) if path else None}
            if state is not None:
                ancestry_available+=1
                rec.update(spatial_stats_from_node_state(state))
                rec.update(gene_state_stats_from_node_state(state))
                if "supernode_id" in state.columns:
                    rec["supernodes_in_state"]=int(state.supernode_id.nunique())
            spatial_rows.append(rec)
    spatial=pd.DataFrame(spatial_rows)

    out=project/"results"/"hierarchy_v104_persistence_basin_landmarks"/sample
    out.mkdir(parents=True,exist_ok=True)
    wpq(basins,out/"persistence_basins.parquet")
    wpq(coh,out/"basin_cross_block_coherence.parquet")
    wpq(detail,out/"basin_block_detail.parquet")
    wpq(landmarks,out/"hierarchy_landmarks.parquet")
    wpq(spatial,out/"landmark_spatial_gene_state.parquet")

    nland=int(landmarks.hierarchy_landmark.sum()) if len(landmarks) else 0
    selected=landmarks[landmarks.hierarchy_landmark.astype(bool)] if len(landmarks) else landmarks
    report={
      "sample":sample,"candidate_minima":int(len(candidates)),"persistence_basins":int(len(basins)),
      "hierarchy_landmarks":nland,
      "landmark_nodes":[int(x) for x in selected.minimum_nodes.tolist()] if len(selected) else [],
      "landmark_removed_fractions":[float(x) for x in selected.minimum_removed_fraction.tolist()] if len(selected) else [],
      "median_lifetime_ell":float(selected.lifetime_ell.median()) if len(selected) else None,
      "median_cross_block_coherence":float(selected.cross_block_coherence.median()) if len(selected) else None,
      "node_state_exports_found":int(ancestry_available),
      "ancestry_spatial_policy":{
        "fabricated_when_missing":False,
        "existing_ledger_only":True,
        "missing_exports_are_reported_not_imputed":True
      },
      "numerical_policy":{
        "empty_support_warns":False,
        "empty_support_returns_unresolved":True,
        "nanmean_used":False
      },
      "status":"PASS" if nland>=1 else "HOLD"
    }
    (out/"persistence_basin_landmark_certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v104_persistence_basin_landmarks.json")
    a=ap.parse_args(); project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())
    c=json.loads(req(project/"results"/"hierarchy_v103_gate_decomposition_metastable_audit"/
                     "gate_decomposition_metastable_global_certificate.json").read_text())
    if c.get("READY_TO_DECIDE_STOPPING_PHILOSOPHY") is not True:
        raise SystemExit("ERROR: v1.0.3 diagnostic gate missing")

    print("STRATA 1.0.4 | Persistence-basin + hierarchy-landmark audit")
    print("No new coarse-graining.")
    print("Local minima are expanded into finite persistence basins.")
    print("Landmark strength combines basin lifetime, basin depth, and cross-block coherence.")
    print("Existing node-state/ancestry exports are attached when actually present.")
    print("Missing lineage/spatial exports are reported, never synthesized.")
    print("Finite-support numerical reductions are warning-free by construction.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result(); reps.append(r)
            print(f"[DONE] {r['sample']}: basins={r['persistence_basins']} "
                  f"landmarks={r['hierarchy_landmarks']} nodes={r['landmark_nodes']} "
                  f"node_states={r['node_state_exports_found']} {r['status']}",flush=True)

    reps.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reps)
    out=project/"results"/"hierarchy_v104_persistence_basin_landmarks"; out.mkdir(parents=True,exist_ok=True)
    g={"strata_version":"1.0.4","stage":"persistence basin and hierarchy landmark audit",
       "sample_reports":reps,
       "PERSISTENCE_BASIN_AUDIT_GATE":"PASS" if gate else "HOLD",
       "HIERARCHY_LANDMARKS_READY":bool(gate),
       "NEW_COARSE_GRAINING_PERFORMED":False,
       "READY_FOR_LANDMARK_AWARE_PRODUCTION_FLOW":bool(gate)}
    p=out/"persistence_basin_landmark_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nPERSISTENCE BASIN AUDIT GATE: {g['PERSISTENCE_BASIN_AUDIT_GATE']}")
    print(f"HIERARCHY LANDMARKS READY: {g['HIERARCHY_LANDMARKS_READY']}")
    print(f"READY FOR LANDMARK-AWARE PRODUCTION FLOW: {g['READY_FOR_LANDMARK_AWARE_PRODUCTION_FLOW']}")
    print(f"Certificate: {p}")
if __name__=="__main__": main()
