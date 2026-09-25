from __future__ import annotations
import argparse,json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from strata_hierarchy.v071.block_preflight import read_10x_cells_by_genes,normalized_expression_dense
from strata_hierarchy.v071.resource_intake import sha256_file
from strata_hierarchy.v074.effective_state import aggregate_supernode_expression
from strata.cli.hierarchy_effective_flow_v0742 import current_relations
from strata_hierarchy.v076.geometry_hierarchy import reconstruct_level0_mechanics,aggregate_mechanics,aggregate_covectors,full_effective_states,geometry_costs
from strata_hierarchy.v090.slow_flow import SlowFlowConfig,choose_slow_batch,apply_contractions,summarize_numeric,shannon_from_sizes,flow_landmark_due
from strata_hierarchy.v090.ledger import FlowLedger
from strata_hierarchy.v091.contextual_flow import ContextualFlowConfig,contextual_block_table,pooled_scales,score_candidates,freeze_cost_thresholds,apply_merge_admissibility,calibration_manifest

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def require(p):
    p=Path(p)
    if not p.exists():raise FileNotFoundError(p)
    return p

def load_static(project,sample):
    l0=project/"results"/"hierarchy_level0_v070"/sample
    v71=project/"results"/"hierarchy_v071_active_blocks"/sample
    v72=project/"results"/"hierarchy_v072_short_pilot"/sample
    v742=project/"results"/"hierarchy_v0742_effective_flow"/sample
    s1=project/"results"/"hierarchy_v075_local_geometry"/"step1_symmetric_base"/sample
    s3=project/"results"/"hierarchy_v075_local_geometry"/"step3_global_bound"/sample
    cells=pd.read_parquet(require(l0/"cells.parquet"))
    edge_rel=pd.read_parquet(require(v72/"level0_candidate_relations.parquet"))
    thresholds=json.loads(require(v72/"frozen_thresholds.json").read_text())
    frozen=json.loads(require(v742/"frozen_effective_thresholds.json").read_text())
    support_floor=float(frozen["communication_support"]["support_floor"])
    fmeta=json.loads(require(v71/"functional_resource_audit.json").read_text())
    lam=float(fmeta["lambda_g"])
    Lfunc=sparse.load_npz(require(v71/"functional_laplacian.npz")).toarray()
    manifest=json.loads(require(l0/"expression_manifest.json").read_text())
    matrix=Path(manifest["source_matrix"])
    if sha256_file(matrix)!=manifest["source_matrix_sha256"]:raise RuntimeError(f"{sample}: expression hash changed")
    X,_,_=read_10x_cells_by_genes(matrix); Y=normalized_expression_dense(X)
    G=sparse.load_npz(require(s1/"symmetric_base_metric.npz")).toarray()
    Bcell=sparse.load_npz(require(s3/"directional_covectors_scaled.npz")).tocsr()
    mech,pressure_audit=reconstruct_level0_mechanics(edge_rel,len(cells))
    return cells,edge_rel,thresholds,support_floor,lam,Lfunc,Y,G,Bcell,mech,pressure_audit

def raw_candidates(edge_rel,labels,node_ids,mol_states,states,Bnode,Lfunc,lam,support_floor,G):
    C0=current_relations(edge_rel,labels,node_ids,mol_states,Lfunc,lam,support_floor,1e-12)
    return geometry_costs(C0,node_ids,states,Bnode,G)

def level0_frame(project_s,sample,cfg_dict):
    os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1")
    project=Path(project_s);cfg=ContextualFlowConfig(**cfg_dict).validate()
    cells,edge_rel,_,support_floor,lam,Lfunc,Y,G,Bcell,mech,_=load_static(project,sample)
    labels=np.arange(len(cells),dtype=np.int64)
    node_ids,mol_states,sizes=aggregate_supernode_expression(Y,labels)
    mech_states=aggregate_mechanics(mech,labels,node_ids); states=full_effective_states(mol_states,mech_states); Bnode=aggregate_covectors(Bcell,labels,node_ids)
    C=raw_candidates(edge_rel,labels,node_ids,mol_states,states,Bnode,Lfunc,lam,support_floor,G)
    F=contextual_block_table(C,node_ids,float(np.mean(sizes)),cfg)
    if len(F)>200000:F=F.iloc[np.linspace(0,len(F)-1,200000,dtype=int)].reset_index(drop=True)
    keep=[c for c in F.columns if c.endswith("_effective") or c.endswith("_reliability_effective")]
    return sample,F[keep].copy()

def slow_cfg(c):
    return SlowFlowConfig(epsilon_fraction=c.epsilon_fraction,soft_step_1=c.soft_step_1,soft_step_2=c.soft_step_2,soft_step_3=c.soft_step_3,rescue_fraction_1=c.rescue_fraction_1,rescue_fraction_2=c.rescue_fraction_2,last_resort_fraction=c.last_resort_fraction,exact_frontier_max_edges=c.exact_frontier_max_edges,checkpoint_every=c.checkpoint_every,heavy_landmark_reduction_fraction=c.heavy_landmark_reduction_fraction,max_microsteps_safety=c.max_microsteps_safety,archive_all_candidates=c.archive_all_candidates).validate()

def one(project_s,sample,cfg_dict,cal):
    os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1")
    t0=time.time(); project=Path(project_s); cfg=ContextualFlowConfig(**cfg_dict).validate(); scfg=slow_cfg(cfg)
    cells,edge_rel,_,support_floor,lam,Lfunc,Y,G,Bcell,mech,pressure_audit=load_static(project,sample)
    labels=np.arange(len(cells),dtype=np.int64); initial_n=len(cells)
    ledger=FlowLedger(project/"results"/"hierarchy_v091_contextual_flow"/"ledger",sample,archive_all_candidates=cfg.archive_all_candidates)
    ledger.checkpoint_labels(0,labels); last_landmark=0.0; total=0; exact=0; rescue=set(); stop=None
    for step in range(cfg.max_microsteps_safety):
        node_ids,mol_states,sizes=aggregate_supernode_expression(Y,labels)
        mech_states=aggregate_mechanics(mech,labels,node_ids); states=full_effective_states(mol_states,mech_states); Bnode=aggregate_covectors(Bcell,labels,node_ids)
        C=raw_candidates(edge_rel,labels,node_ids,mol_states,states,Bnode,Lfunc,lam,support_floor,G)
        mean_size=float(np.mean(sizes)) if len(sizes) else 1.0
        F=contextual_block_table(C,node_ids,mean_size,cfg); S=score_candidates(F,cal["scales"],cfg); A=apply_merge_admissibility(S,mean_size,cal["thresholds"],cfg)
        decision=choose_slow_batch(A,len(node_ids),step,scfg); elig=decision["eligible"]; selected=decision["selected"]; ledger.write_candidates(step,A)
        row={"sample":sample,"microstep":int(step),"nodes_before":int(len(node_ids)),"level0_cells":int(initial_n),"cumulative_merges":int(total),"removed_fraction_before":float((initial_n-len(node_ids))/initial_n),"superedges":int(len(A)),"merge_allowed":int(A.merge_allowed.sum()) if len(A) else 0,"selection_mode":decision["selection_mode"],"step_mode":decision["step_mode"],"batch_cap":int(decision["batch_cap"]),"routes_considered":decision["routes_considered"],"selected_merges":int(len(selected)),"supernode_size_entropy":shannon_from_sizes(sizes),"supernode_size_mean":mean_size,"context_hops":int(A.context_hops.iloc[0]) if len(A) else 0,"merge_cost_threshold":float(A.merge_cost_threshold.iloc[0]) if len(A) else np.nan}
        for col,prefix in [("composite_merge_cost","merge_cost"),("evidence_coverage","evidence"),("mechanics_reliability_effective","mech_reliability"),("geometry_reliability_effective","geom_reliability")]:
            if col in A:row.update(summarize_numeric(A[col],prefix))
        if len(elig)==0:
            stop="no_contextually_admissible_boundaries";row["stop_reason"]=stop;ledger.append_step(row);break
        if len(selected)==0:raise RuntimeError(f"{sample}: eligible boundaries remain but no selection")
        if decision["exact_frontier"]:exact+=1
        if decision["step_mode"]!="epsilon":rescue.add(decision["step_mode"])
        selected=selected.copy();selected["survivor_node"]=np.minimum(selected.super_i.astype(np.int64),selected.super_j.astype(np.int64));selected["removed_node"]=np.maximum(selected.super_i.astype(np.int64),selected.super_j.astype(np.int64));ledger.write_merges(step,selected)
        labels=apply_contractions(labels,selected);total+=len(selected);after_n=len(np.unique(labels));after_removed=(initial_n-after_n)/initial_n
        row["nodes_after"]=int(after_n);row["removed_fraction_after"]=float(after_removed);row["cumulative_merges_after"]=int(total)
        due,new_landmark=flow_landmark_due(after_removed,last_landmark,scfg)
        if due:
            ledger.write_heavy_marker(step,{"sample":sample,"microstep":int(step),"removed_fraction":float(after_removed),"landmark_fraction":float(new_landmark),"nodes":int(after_n),"replay_checkpoint_available":True,"requested_heavy_observables":["transport_component_structure","directed_geodesic_landmarks","transport_holonomy","holonomy_density","functional_pathway_state","mechanics_state","communication_state","contextual_block_state"]});last_landmark=new_landmark
        if step%cfg.checkpoint_every==0:ledger.checkpoint_labels(step+1,labels)
        ledger.append_step(row)
    else:stop="microstep_safety_ceiling_with_unknown_remaining_frontier"
    ledger.checkpoint_labels(step+1,labels);final_nodes=len(np.unique(labels));natural=(stop=="no_contextually_admissible_boundaries")
    summary={"sample":sample,"level0_cells":int(initial_n),"final_nodes":int(final_nodes),"total_merges":int(total),"removed_fraction":float((initial_n-final_nodes)/initial_n),"microsteps":int(step+1),"stop_reason":stop,"natural_exhaustion":bool(natural),"exact_frontier_steps":int(exact),"rescue_modes_used":sorted(rescue),"pressure_audit_source":pressure_audit,"elapsed_seconds":float(time.time()-t0),"ledger_root":str(ledger.root),"status":"PASS" if natural else "HOLD"}
    (ledger.root/"flow_summary.json").write_text(json.dumps(summary,indent=2)+"\n");return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--project-root",default=".");ap.add_argument("--workers",type=int,default=3);ap.add_argument("--config",default="configs/hierarchy_v091_contextual_flow.json");ap.add_argument("--calibrate-only",action="store_true");a=ap.parse_args()
    project=Path(a.project_root).resolve()
    atlas=require(project/"results"/"hierarchy_v079_loop_normalized_geometry"/"loop_normalized_geometry_global_certificate.json")
    if json.loads(atlas.read_text()).get("READY_FOR_SPATIAL_GEOMETRY_STATISTICS") is not True:raise SystemExit("v0.7.9 not certified")
    cfg_dict=json.loads(require(project/a.config).read_text())["contextual_flow"];cfg=ContextualFlowConfig(**cfg_dict).validate()
    print("SUTRA / STRATA 0.9.1 | Contextual convolution effective flow")
    print("Missing local mechanics, zero communication, and unresolved local geometry are reliability states, not physical walls.")
    frames=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(SAMPLES))) as ex:
        futs={ex.submit(level0_frame,str(project),s,cfg_dict):s for s in SAMPLES}
        for f in as_completed(futs):
            s,fr=f.result();frames.append(fr);print(f"[CAL] {s}: {len(fr):,}",flush=True)
    scales=pooled_scales(frames);scored=[score_candidates(f,scales,cfg) for f in frames];thresholds=freeze_cost_thresholds(scored,cfg);cal={"scales":scales,"thresholds":thresholds}
    out=project/"results"/"hierarchy_v091_contextual_flow";out.mkdir(parents=True,exist_ok=True)
    manifest=calibration_manifest(scales,thresholds,cfg);manifest["source_v079_certificate_sha256"]=sha256_file(atlas);(out/"contextual_calibration.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Frozen pooled q{thresholds['initial_quantile']:.2f} threshold={thresholds['initial_threshold']:.6g}; cap q{thresholds['maximum_quantile']:.2f}={thresholds['maximum_threshold']:.6g}")
    if a.calibrate_only:return 0
    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(SAMPLES))) as ex:
        futs={ex.submit(one,str(project),s,cfg_dict,cal):s for s in SAMPLES}
        for f in as_completed(futs):
            r=f.result();reports.append(r);print(f"[DONE] {r['sample']}: {r['level0_cells']:,}->{r['final_nodes']:,} removed={100*r['removed_fraction']:.2f}% {r['status']}",flush=True)
    reports.sort(key=lambda x:SAMPLES.index(x["sample"]));gate=all(r["status"]=="PASS" for r in reports)
    cert={"strata_version":"0.9.1","public_name":"SUTRA","stage":"contextual convolution effective flow","source_v079_certificate_sha256":sha256_file(atlas),"scientific_policy":{"contextual_graph_convolution":True,"scale_adaptive_context_radius":True,"hard_local_mechanics_veto":False,"hard_zero_communication_veto":False,"hard_unresolved_geometry_veto":False,"smooth_noncompensatory_aggregation":"weighted log-sum-exp","shared_level0_calibration_across_specimens":True,"later_level_rescaling":False,"target_final_node_count":None},"calibration":manifest,"sample_reports":reports,"CONTEXTUAL_FLOW_GATE":"PASS" if gate else "HOLD","FLOW_LEDGER_COMPLETE":bool(gate)}
    p=out/"contextual_flow_global_certificate.json";p.write_text(json.dumps(cert,indent=2)+"\n");print(f"Certificate: {p}");return 0
if __name__=="__main__":raise SystemExit(main())
