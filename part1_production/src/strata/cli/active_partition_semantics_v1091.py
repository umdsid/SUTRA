from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow as pa,pyarrow.parquet as pq

from strata_hierarchy.v1091.semantics import *

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def req(p):
    p=Path(p)
    if not p.exists():raise FileNotFoundError(p)
    return p

def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def state_path(manifest,candidate):
    q=manifest[pd.to_numeric(manifest.candidate_landmark,errors="coerce")==int(candidate)]
    if len(q)==0:return None
    r=q.iloc[0]
    p=r.get("node_state_path",r.get("node_state_source",None))
    return Path(str(p)) if p is not None and not (isinstance(p,float) and np.isnan(p)) else None

def one(project_s,sample,cfg):
    project=Path(project_s)
    v1061=project/"results"/"hierarchy_v1061_full_landmark_materialization"/sample
    table=pd.read_parquet(req(v1061/"landmark_table.parquet")).sort_values("minimum_removed_fraction").reset_index(drop=True)
    manifest=pd.read_parquet(req(
        project/"results"/"hierarchy_v104_persistence_basin_landmarks"/sample/
        "landmark_spatial_gene_state.parquet"
    ))

    # Expected conserved Level-0 mass from v1.0.9, which already found it constant.
    v109=pd.read_parquet(req(
        project/"results"/"hierarchy_v109_lineage_conservation_domain_coherence"/sample/
        "landmark_partition_audit.parquet"
    ))
    vals=pd.to_numeric(v109.unique_level0_members,errors="coerce")
    vals=vals[np.isfinite(vals)]
    expected_mass=int(vals.iloc[0]) if len(vals) and vals.nunique()==1 else None

    schema_rows=[];eval_parts=[];choice_rows=[];active_states={}
    for _,lm in table.iterrows():
        lid=str(lm.landmark_id)
        p=state_path(manifest,int(lm.candidate_landmark))
        if p is None or not p.exists():
            choice_rows.append({"landmark_id":lid,"resolved":False,"reason":"state_path_missing"})
            continue
        d=pd.read_parquet(p)
        schema_rows.append({"landmark_id":lid,**state_schema_record(p,d)})

        target_step=None
        for c in ["microstep","step","minimum_step","candidate_landmark"]:
            if c in lm.index and pd.notna(lm[c]):
                try:target_step=int(lm[c]);break
                except Exception:pass
        target_level=None
        if "landmark_order" in lm.index and pd.notna(lm.landmark_order):
            target_level=int(lm.landmark_order)

        e=evaluate_masks(
            d,expected_nodes=int(lm.minimum_nodes),
            expected_level0_mass=expected_mass,
            target_step=target_step,target_level=target_level
        )
        e.insert(0,"landmark_id",lid)
        eval_parts.append(e)
        ch=choose_unique_partition(e)
        if ch is None:
            n_exact=int(e.exact_partition.sum()) if len(e) else 0
            choice_rows.append({
                "landmark_id":lid,"resolved":False,
                "reason":"no_unique_exact_active_partition",
                "exact_candidate_count":n_exact
            })
        else:
            active=apply_named_candidate(d,ch["candidate"],target_step,target_level)
            active_states[lid]=active
            choice_rows.append({
                "landmark_id":lid,"resolved":True,
                "reason":"unique_exact_active_partition",
                "active_candidate":ch["candidate"],
                "candidate_kind":ch["candidate_kind"],
                "active_rows":int(len(active)),
                "expected_nodes":int(lm.minimum_nodes),
                "unique_level0_members":expected_mass
            })

    schemas=pd.DataFrame(schema_rows)
    evals=pd.concat(eval_parts,ignore_index=True) if eval_parts else pd.DataFrame()
    choices=pd.DataFrame(choice_rows)

    # Nesting only after active partition reconstruction.
    lineage_parts=[];pair_rows=[]
    for i in range(len(table)-1):
        a=str(table.iloc[i].landmark_id);b=str(table.iloc[i+1].landmark_id)
        if a not in active_states or b not in active_states:
            pair_rows.append({"child_landmark_id":a,"parent_landmark_id":b,"resolved":False,"nested":False})
            continue
        child=map_partition(active_states[a]);parent=map_partition(active_states[b])
        d,ok=compare_nested(child,parent)
        if len(d):
            d.insert(0,"child_landmark_id",a);d.insert(1,"parent_landmark_id",b);lineage_parts.append(d)
        pair_rows.append({"child_landmark_id":a,"parent_landmark_id":b,"resolved":True,"nested":bool(ok)})
    lineage=pd.concat(lineage_parts,ignore_index=True) if lineage_parts else pd.DataFrame()
    pairs=pd.DataFrame(pair_rows)

    all_active=bool(len(choices)==len(table) and choices.resolved.fillna(False).all())
    nested_all=bool(len(pairs)==max(len(table)-1,0) and len(pairs) and pairs.resolved.all() and pairs.nested.all()) if len(table)>1 else all_active
    semantics_certified=bool(all_active and nested_all)

    inv=classify_source_inventory(inventory_level0_sources(project,cfg))
    out=project/"results"/"hierarchy_v1091_active_partition_semantics"/sample
    out.mkdir(parents=True,exist_ok=True)
    wpq(schemas,out/"state_schema_inventory.parquet")
    wpq(evals,out/"active_partition_candidate_audit.parquet")
    wpq(choices,out/"active_partition_choices.parquet")
    wpq(pairs,out/"active_partition_pair_audit.parquet")
    wpq(lineage,out/"active_supernode_lineage_nesting.parquet")
    wpq(inv,out/"level0_source_inventory.parquet")

    # Persist resolved active partitions for the next coherence stage.
    active_dir=out/"active_partitions"
    active_dir.mkdir(exist_ok=True)
    for lid,d in active_states.items():
        safe=lid.replace(":","__").replace("/","_")
        wpq(d,active_dir/f"{safe}.parquet")

    top_expr=[]
    top_coord=[]
    if len(inv):
        top_expr=inv.sort_values(["expression_score","size_bytes"],ascending=[False,False]).head(10)[
            ["relative_path","suffix","expression_score","coordinate_score","size_bytes"]
        ].to_dict("records")
        top_coord=inv.sort_values(["coordinate_score","size_bytes"],ascending=[False,False]).head(10)[
            ["relative_path","suffix","expression_score","coordinate_score","size_bytes"]
        ].to_dict("records")

    cert={
        "sample":sample,
        "landmarks":int(len(table)),
        "expected_level0_mass":expected_mass,
        "active_partitions_resolved":int(choices.resolved.fillna(False).sum()) if len(choices) else 0,
        "active_partitions_expected":int(len(table)),
        "all_active_partitions_resolved":all_active,
        "nested_active_partitions_all_pairs":nested_all,
        "active_partition_semantics_certified":semantics_certified,
        "top_expression_source_candidates":top_expr,
        "top_coordinate_source_candidates":top_coord,
        "scientific_hierarchy_changed":False,
        "new_coarse_graining_performed":False,
        "status":"PASS" if semantics_certified else "HOLD"
    }
    (out/"active_partition_semantics_certificate.json").write_text(json.dumps(cert,indent=2)+"\n")
    return cert

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v1091_active_partition_semantics.json")
    a=ap.parse_args();project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())

    c=json.loads(req(
        project/"results"/"hierarchy_v109_lineage_conservation_domain_coherence"/
        "lineage_conservation_domain_coherence_global_certificate.json"
    ).read_text())
    if c.get("SCIENTIFIC_HIERARCHY_CHANGED") is not False:
        raise SystemExit("ERROR: v1.0.9 hierarchy provenance unexpected")

    print("STRATA 1.0.9.1 | Active-partition semantics + Level-0 source audit")
    print("No new coarse-graining and no hierarchy changes.")
    print("Active rows are reconstructed only from semantics explicitly present in each state schema.")
    print("A candidate is accepted only if it uniquely matches expected node count, conserved Level-0 mass and disjoint partition semantics.")
    print("Level-0 expression/coordinate stores are inventoried across Parquet, HDF5/H5AD, NPZ and CSV-like resources.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result();reps.append(r)
            print(
                f"[DONE] {r['sample']}: active={r['active_partitions_resolved']}/{r['active_partitions_expected']} "
                f"nested={r['nested_active_partitions_all_pairs']} {r['status']}",flush=True
            )

    reps.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["active_partition_semantics_certified"] for r in reps)
    out=project/"results"/"hierarchy_v1091_active_partition_semantics"
    out.mkdir(parents=True,exist_ok=True)
    g={
        "strata_version":"1.0.9.1",
        "stage":"active partition semantics and Level-0 source audit",
        "sample_reports":reps,
        "ACTIVE_PARTITION_SEMANTICS_GATE":"PASS" if gate else "HOLD",
        "ACTIVE_PARTITION_SEMANTICS_CERTIFIED":bool(gate),
        "READY_FOR_DOMINANT_DOMAIN_COHERENCE_REPLAY":bool(gate),
        "SCIENTIFIC_HIERARCHY_CHANGED":False,
        "NEW_COARSE_GRAINING_PERFORMED":False
    }
    p=out/"active_partition_semantics_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nACTIVE PARTITION SEMANTICS GATE: {g['ACTIVE_PARTITION_SEMANTICS_GATE']}")
    print(f"ACTIVE PARTITION SEMANTICS CERTIFIED: {g['ACTIVE_PARTITION_SEMANTICS_CERTIFIED']}")
    print(f"READY FOR DOMINANT DOMAIN COHERENCE REPLAY: {g['READY_FOR_DOMINANT_DOMAIN_COHERENCE_REPLAY']}")
    print("SCIENTIFIC HIERARCHY CHANGED: False")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
