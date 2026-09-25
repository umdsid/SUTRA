from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from sutra.hierarchy.v106.materialize import *

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")
def req(p):
    p=Path(p)
    if not p.exists():raise FileNotFoundError(p)
    return p
def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def one(project_s,sample,cfg):
    project=Path(project_s)
    v105=project/"results"/"hierarchy_v105_basin_failure_pareto_audit"/sample
    v103=project/"results"/"hierarchy_v103_gate_decomposition_metastable_audit"/sample
    v94=project/"results"/"hierarchy_v094_ness_flow_diagnostics"/sample

    front=pd.read_parquet(req(v105/"pareto_front_basins.parquet"))
    all_layers=pd.read_parquet(req(v105/"pareto_hierarchy_layers.parquet"))
    gates=pd.read_parquet(req(v103/"gate_decomposition_by_landmark.parquet"))
    flow=pd.read_parquet(req(v94/"landmark_flow_diagnostics.parquet"))

    front=compute_lsi(assign_landmark_ids(front,sample),cfg)
    if not validate_order(front):raise RuntimeError(f"{sample}: invalid landmark order")
    tree=build_hierarchy_tree(front,sample)
    transitions=build_transition_table(front)

    obs=[];states=[];lins=[]
    for r in front.itertuples(index=False):
        lm=int(r.candidate_landmark);lid=str(r.landmark_id)
        hit=gates[gates.landmark_index.astype(int)==lm]
        if len(hit):obs.append(materialize_observable_row(hit.iloc[0].to_dict(),lid))
        else:
            hit=flow[flow.landmark_index.astype(int)==lm]
            obs.append(materialize_observable_row(hit.iloc[0].to_dict(),lid) if len(hit) else {"landmark_id":lid})
        state,path=load_best_node_state(project,sample,lm)
        states.append(summarize_node_state(state,lid,path))
        lin=extract_lineage(state,lid)
        if len(lin):lins.append(lin)

    observables=pd.DataFrame(obs)
    state_summary=pd.DataFrame(states)
    lineage=pd.concat(lins,ignore_index=True) if lins else pd.DataFrame(
        columns=["landmark_id","supernode_id","level0_members","parent_supernode_id"]
    )
    groups=observable_groups(observables.columns)

    keep=[c for c in [
        "landmark_id","landmark_order","principal_landmark","pareto_layer",
        "candidate_landmark","entry_landmark","exit_landmark","entry_nodes","minimum_nodes","exit_nodes",
        "entry_removed_fraction","minimum_removed_fraction","exit_removed_fraction",
        "entry_ell","minimum_ell","exit_ell","lifetime_ell","basin_landmarks",
        "minimum_score","shoulder_level","basin_depth","cross_block_coherence",
        "unstable_block_switch_entropy","stable_blocks","total_blocks","active_blocks",
        "lsi","lsi_terms_resolved","lsi_is_gate"
    ] if c in front.columns]
    table=front[keep].copy()

    out=project/"results"/"hierarchy_v106_frozen_landmark_materialization"/sample
    out.mkdir(parents=True,exist_ok=True)
    wpq(table,out/"landmark_table.parquet")
    wpq(tree,out/"landmark_tree.parquet")
    wpq(transitions,out/"landmark_transitions.parquet")
    wpq(observables,out/"landmark_observables.parquet")
    wpq(state_summary,out/"landmark_spatial_gene_state.parquet")
    wpq(lineage,out/"landmark_lineage.parquet")
    wpq(all_layers,out/"pareto_hierarchy_all_layers.parquet")
    (out/"hierarchy_graph.graphml").write_text(graphml_text(tree))
    for g,cols in groups.items():
        cols=[c for c in cols if c in observables.columns]
        if cols:wpq(observables[["landmark_id"]+cols],out/f"landmark_{g}.parquet")

    nstate=int(state_summary.node_state_available.sum()) if len(state_summary) else 0
    cert={
        "sample":sample,
        "landmark_definition":"Pareto-supported persistence basin; Pareto layer 1 = principal landmark",
        "weighted_scalarization_used_for_existence":False,
        "absolute_cross_block_coherence_gate_used_for_existence":False,
        "lsi_used_for_existence":False,
        "principal_landmarks":int(len(front)),
        "principal_landmark_nodes":[int(x) for x in front.minimum_nodes.tolist()],
        "hierarchy_order_valid":bool(validate_order(front)),
        "node_state_exports_found":nstate,
        "lineage_available":bool(len(lineage)),
        "lineage_rows":int(len(lineage)),
        "missing_data_fabricated":False,
        "new_coarse_graining_performed":False,
        "observable_groups":{k:int(len(v)) for k,v in groups.items()},
        "status":"PASS" if len(front)>=1 and validate_order(front) else "HOLD",
    }
    (out/"hierarchy_certificate.json").write_text(json.dumps(cert,indent=2)+"\n")
    (out/"landmark_summary.json").write_text(json.dumps({
        "sample":sample,
        "principal_landmarks":cert["principal_landmarks"],
        "principal_landmark_nodes":cert["principal_landmark_nodes"],
        "state_exports_found":nstate,
        "lineage_rows":int(len(lineage)),
        "lsi_descriptive_only":True,
        "all_pareto_layers_archived":True,
        "missing_state_policy":"report_missing_never_synthesize"
    },indent=2)+"\n")
    return cert

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v106_frozen_landmark_materialization.json")
    a=ap.parse_args();project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())
    c=json.loads(req(project/"results"/"hierarchy_v105_basin_failure_pareto_audit"/"basin_failure_pareto_global_certificate.json").read_text())
    if c.get("PARETO_BASIN_LANDSCAPE_READY") is not True:
        raise SystemExit("ERROR: v1.0.5 Pareto landscape not ready")

    print("STRATA 1.0.6 | Frozen Pareto-landmark materialization")
    print("No new coarse-graining.")
    print("Principal landmark existence = Pareto layer-1 persistence basin.")
    print("No absolute coherence threshold or weighted scalarization defines existence.")
    print("LSI is descriptive only. All Pareto layers are archived.")
    print("Existing state/lineage exports are attached when present; missing data are never fabricated.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result();reps.append(r)
            print(f"[DONE] {r['sample']}: landmarks={r['principal_landmarks']} "
                  f"nodes={r['principal_landmark_nodes']} states={r['node_state_exports_found']} "
                  f"lineage={r['lineage_available']} {r['status']}",flush=True)

    reps.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reps)
    out=project/"results"/"hierarchy_v106_frozen_landmark_materialization"
    out.mkdir(parents=True,exist_ok=True)
    g={
        "strata_version":"1.0.6","stage":"frozen Pareto landmark materialization",
        "sample_reports":reps,
        "FROZEN_LANDMARK_MATERIALIZATION_GATE":"PASS" if gate else "HOLD",
        "PARETO_LANDMARK_DEFINITION_FROZEN":bool(gate),
        "LANDMARK_HIERARCHIES_MATERIALIZED":bool(gate),
        "LSI_IS_DESCRIPTIVE_ONLY":True,
        "NEW_COARSE_GRAINING_PERFORMED":False,
        "READY_FOR_LANDMARK_AWARE_PRODUCTION_FLOW":bool(gate),
        "READY_FOR_CROSS_SPECIMEN_HIERARCHY_ATLAS":bool(gate)
    }
    p=out/"frozen_landmark_materialization_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nFROZEN LANDMARK MATERIALIZATION GATE: {g['FROZEN_LANDMARK_MATERIALIZATION_GATE']}")
    print(f"PARETO LANDMARK DEFINITION FROZEN: {g['PARETO_LANDMARK_DEFINITION_FROZEN']}")
    print(f"LANDMARK HIERARCHIES MATERIALIZED: {g['LANDMARK_HIERARCHIES_MATERIALIZED']}")
    print("LSI IS DESCRIPTIVE ONLY: True")
    print(f"READY FOR LANDMARK-AWARE PRODUCTION FLOW: {g['READY_FOR_LANDMARK_AWARE_PRODUCTION_FLOW']}")
    print(f"READY FOR CROSS-SPECIMEN HIERARCHY ATLAS: {g['READY_FOR_CROSS_SPECIMEN_HIERARCHY_ATLAS']}")
    print(f"Certificate: {p}")
if __name__=="__main__":main()
