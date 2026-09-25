from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from sutra.hierarchy.v1061.resolve import *

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")
def req(p):
    p=Path(p)
    if not p.exists():raise FileNotFoundError(p)
    return p
def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def one(project_s,sample,cfg):
    project=Path(project_s)
    v106=project/"results"/"hierarchy_v106_frozen_landmark_materialization"/sample
    table=pd.read_parquet(req(v106/"landmark_table.parquet"))
    tree=pd.read_parquet(req(v106/"landmark_tree.parquet"))
    transitions=pd.read_parquet(req(v106/"landmark_transitions.parquet"))
    all_layers=pd.read_parquet(req(v106/"pareto_hierarchy_all_layers.parquet"))

    inv=scan_parquets(project,sample,cfg)
    manifest=v104_state_manifest(project,sample)

    provenance_parts=[];wide_rows=[];state_rows=[];lineage_parts=[]
    for _,lr in table.iterrows():
        lm=lr.to_dict()
        p,w=materialize_landmark_groups(inv,lm)
        provenance_parts.append(p);wide_rows.append(w)

        sp=state_path_for_candidate(manifest,int(lm["candidate_landmark"]))
        state=read_state(sp)
        state_rows.append(state_summary(state,str(lm["landmark_id"]),sp))
        lin=lineage_from_state(state,str(lm["landmark_id"]))
        if len(lin):lineage_parts.append(lin)

    prov=pd.concat(provenance_parts,ignore_index=True) if provenance_parts else pd.DataFrame()
    wide=pd.DataFrame(wide_rows)
    states=pd.DataFrame(state_rows)
    lineage=pd.concat(lineage_parts,ignore_index=True) if lineage_parts else pd.DataFrame(
        columns=["landmark_id","supernode_id","level0_members","parent_supernode_id"]
    )
    matrix=completeness_matrix(prov)
    complete=sample_complete(prov,states,cfg)

    out=project/"results"/"hierarchy_v1061_full_landmark_materialization"/sample
    out.mkdir(parents=True,exist_ok=True)
    wpq(table,out/"landmark_table.parquet")
    wpq(tree,out/"landmark_tree.parquet")
    wpq(transitions,out/"landmark_transitions.parquet")
    wpq(all_layers,out/"pareto_hierarchy_all_layers.parquet")
    wpq(inv,out/"source_parquet_inventory.parquet")
    wpq(prov,out/"landmark_source_provenance.parquet")
    wpq(matrix,out/"landmark_completeness_matrix.parquet")
    wpq(wide,out/"landmark_observables_full.parquet")
    wpq(states,out/"landmark_spatial_gene_state.parquet")
    wpq(lineage,out/"landmark_lineage.parquet")

    # Per-group tables with their resolved columns.
    for group in GROUPS:
        gprov=prov[(prov.group==group)&prov.resolved.astype(bool)]
        if len(gprov)==0:continue
        cols=["landmark_id"]
        for lid in gprov.landmark_id:
            pass
        group_cols=[c for c in wide.columns if c!="landmark_id" and any(c.startswith(p) for p in GROUPS[group])]
        if group_cols:
            wpq(wide[["landmark_id"]+group_cols],out/f"landmark_{group}.parquet")

    # Reuse v1.0.6 graph; scientific hierarchy unchanged.
    src_graph=v106/"hierarchy_graph.graphml"
    if src_graph.exists():
        (out/"hierarchy_graph.graphml").write_text(src_graph.read_text())

    counts={g:int(((prov.group==g)&prov.resolved.astype(bool)).sum()) for g in GROUPS}
    total_landmarks=int(len(table))
    state_found=int(states.node_state_available.sum()) if len(states) else 0
    cert={
        "sample":sample,
        "principal_landmarks":total_landmarks,
        "principal_landmark_nodes":[int(x) for x in table.minimum_nodes.tolist()],
        "scientific_hierarchy_changed":False,
        "pareto_landmark_definition_changed":False,
        "observable_group_resolved_landmarks":counts,
        "node_state_exports_found":state_found,
        "node_state_expected":total_landmarks,
        "lineage_rows":int(len(lineage)),
        "lineage_available":bool(len(lineage)),
        "all_required_groups_resolved":bool(
            all(counts[g]==total_landmarks for g in cfg["required_observable_groups"])
        ),
        "all_node_states_resolved":bool(state_found==total_landmarks),
        "full_materialization_complete":bool(complete),
        "missing_data_fabricated":False,
        "state_resolution_uses_v104_manifest":True,
        "observable_resolution_exact_keys_only":True,
        "status":"PASS" if complete else "HOLD",
    }
    (out/"hierarchy_certificate.json").write_text(json.dumps(cert,indent=2)+"\n")
    return cert

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v1061_state_resolution.json")
    a=ap.parse_args();project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())

    c=json.loads(req(project/"results"/"hierarchy_v106_frozen_landmark_materialization"/"frozen_landmark_materialization_global_certificate.json").read_text())
    if c.get("PARETO_LANDMARK_DEFINITION_FROZEN") is not True:
        raise SystemExit("ERROR: v1.0.6 landmark definition not frozen")

    print("STRATA 1.0.6.1 | State-resolution + full landmark materialization repair")
    print("Scientific hierarchy is unchanged.")
    print("Node states are resolved from the exact v1.0.4 source manifest.")
    print("Observable sources are discovered by schema and joined by exact scale keys only.")
    print("Every block carries explicit source provenance.")
    print("The gate PASSes only if every required block and node state resolves for every principal landmark.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result();reps.append(r)
            cc=r["observable_group_resolved_landmarks"]
            print(f"[DONE] {r['sample']}: landmarks={r['principal_landmarks']} "
                  f"states={r['node_state_exports_found']}/{r['node_state_expected']} "
                  f"expr={cc['expression']} func={cc['functional']} chat={cc['cellchat']} "
                  f"mech={cc['mechanics']} press={cc['pressure']} topo={cc['topology']} "
                  f"geom={cc['directional_geometry']} trans={cc['transport']} "
                  f"geo={cc['geodesics']} holo={cc['holonomy']} deriv={cc['derivatives']} "
                  f"{r['status']}",flush=True)

    reps.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reps)
    out=project/"results"/"hierarchy_v1061_full_landmark_materialization"
    out.mkdir(parents=True,exist_ok=True)
    g={
        "strata_version":"1.0.6.1",
        "stage":"state resolution and full frozen-landmark materialization",
        "sample_reports":reps,
        "SCIENTIFIC_HIERARCHY_CHANGED":False,
        "PARETO_LANDMARK_DEFINITION_CHANGED":False,
        "FULL_LANDMARK_MATERIALIZATION_GATE":"PASS" if gate else "HOLD",
        "FULL_LANDMARK_STATE_MATERIALIZED":bool(gate),
        "READY_FOR_LANDMARK_AWARE_PRODUCTION_FLOW":bool(gate),
        "READY_FOR_CROSS_SPECIMEN_HIERARCHY_ATLAS":bool(gate)
    }
    p=out/"full_landmark_materialization_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nFULL LANDMARK MATERIALIZATION GATE: {g['FULL_LANDMARK_MATERIALIZATION_GATE']}")
    print("SCIENTIFIC HIERARCHY CHANGED: False")
    print(f"FULL LANDMARK STATE MATERIALIZED: {g['FULL_LANDMARK_STATE_MATERIALIZED']}")
    print(f"READY FOR LANDMARK-AWARE PRODUCTION FLOW: {g['READY_FOR_LANDMARK_AWARE_PRODUCTION_FLOW']}")
    print(f"READY FOR CROSS-SPECIMEN HIERARCHY ATLAS: {g['READY_FOR_CROSS_SPECIMEN_HIERARCHY_ATLAS']}")
    print(f"Certificate: {p}")
if __name__=="__main__":main()
