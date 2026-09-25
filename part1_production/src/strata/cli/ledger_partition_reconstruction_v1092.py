from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow as pa,pyarrow.parquet as pq

from strata_hierarchy.v1092.reconstruct import *

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")
def req(p):
    p=Path(p)
    if not p.exists():raise FileNotFoundError(p)
    return p
def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def one(project_s,sample,cfg):
    project=Path(project_s)
    table=pd.read_parquet(req(
        project/"results"/"hierarchy_v1061_full_landmark_materialization"/sample/"landmark_table.parquet"
    )).sort_values("minimum_removed_fraction").reset_index(drop=True)

    n0,n0src,n0method=derive_level0_count(project,sample)
    if n0 is None:raise RuntimeError(f"{sample}: cannot derive Level-0 cell count")

    audit_parts=[];choice_rows=[];maps={};label_vectors={}
    for _,lm in table.iterrows():
        lid=str(lm.landmark_id); expected=int(lm.minimum_nodes)
        aud,payloads=scan_assignment_candidates(project,sample,n0,expected,cfg)
        if len(aud):aud.insert(0,"landmark_id",lid);audit_parts.append(aud)
        ch=choose_exact_candidate(aud)
        if ch is None:
            choice_rows.append({"landmark_id":lid,"resolved":False,
                                "exact_candidates":int(aud.exact_node_match.sum()) if len(aud) else 0})
            continue
        key=(ch["path"],ch["array_key"])
        obj=payloads[key]
        if isinstance(obj,np.ndarray):
            m=membership_map_from_label_vector(obj)
            label_vectors[lid]=obj
        else:
            m=membership_map_from_assignment_df(obj)
        inv=partition_invariants(m)
        exact=bool(inv["nodes"]==expected and inv["unique"]==n0 and inv["total"]==n0 and inv["disjoint"])
        if exact:maps[lid]=m
        choice_rows.append({
            "landmark_id":lid,"resolved":exact,"path":ch["path"],"kind":ch["kind"],
            "array_key":ch["array_key"],"expected_nodes":expected,
            "nodes":inv["nodes"],"unique_level0":inv["unique"],
            "total_assignments":inv["total"],"duplicates":inv["duplicates"]
        })

    audits=pd.concat(audit_parts,ignore_index=True) if audit_parts else pd.DataFrame()
    choices=pd.DataFrame(choice_rows)

    lineage_parts=[];pair_rows=[]
    for i in range(len(table)-1):
        a=str(table.iloc[i].landmark_id);b=str(table.iloc[i+1].landmark_id)
        if a not in maps or b not in maps:
            pair_rows.append({"child_landmark_id":a,"parent_landmark_id":b,"resolved":False,"nested":False})
            continue
        d,ok=compare_nested(maps[a],maps[b])
        if len(d):
            d.insert(0,"child_landmark_id",a);d.insert(1,"parent_landmark_id",b);lineage_parts.append(d)
        pair_rows.append({"child_landmark_id":a,"parent_landmark_id":b,"resolved":True,"nested":ok})
    lineage=pd.concat(lineage_parts,ignore_index=True) if lineage_parts else pd.DataFrame()
    pairs=pd.DataFrame(pair_rows)

    all_parts=bool(len(choices)==len(table) and choices.resolved.fillna(False).all())
    nested=bool(len(pairs) and pairs.resolved.all() and pairs.nested.all()) if len(table)>1 else all_parts
    lineage_ok=bool(all_parts and nested)

    # Coherence on exact label-vector checkpoints when available.
    X,genes,expr_src=load_xenium_expression(project,sample)
    coords,coord_src=load_coordinates(project,sample,n0)
    expr_parts=[];sp_parts=[]
    for lid,lab in label_vectors.items():
        e=expression_coherence(lab,X)
        if len(e):e.insert(0,"landmark_id",lid);expr_parts.append(e)
        s=spatial_coherence(lab,coords)
        if len(s):s.insert(0,"landmark_id",lid);sp_parts.append(s)
    expr=pd.concat(expr_parts,ignore_index=True) if expr_parts else pd.DataFrame()
    spatial=pd.concat(sp_parts,ignore_index=True) if sp_parts else pd.DataFrame()

    out=project/"results"/"hierarchy_v1092_ledger_partition_reconstruction"/sample
    out.mkdir(parents=True,exist_ok=True)
    wpq(audits,out/"assignment_candidate_audit.parquet")
    wpq(choices,out/"landmark_partition_choices.parquet")
    wpq(pairs,out/"landmark_pair_nesting.parquet")
    wpq(lineage,out/"supernode_lineage_nesting.parquet")
    wpq(expr,out/"supernode_expression_coherence.parquet")
    wpq(spatial,out/"supernode_spatial_coherence.parquet")
    # save exact label arrays for reproducibility
    ldir=out/"labels";ldir.mkdir(exist_ok=True)
    for lid,lab in label_vectors.items():
        np.savez_compressed(ldir/f"{lid.replace(':','__')}.npz",labels=lab)

    cert={
        "sample":sample,"level0_cells":int(n0),"level0_source":n0src,"level0_count_method":n0method,
        "landmarks":int(len(table)),
        "partitions_resolved":int(choices.resolved.fillna(False).sum()),
        "all_partitions_resolved":all_parts,
        "nested_lineage_all_pairs":nested,
        "lineage_mass_conservation_certified":lineage_ok,
        "expression_source":expr_src,
        "coordinate_source":coord_src,
        "expression_coherence_landmarks":int(expr.landmark_id.nunique()) if len(expr) else 0,
        "spatial_coherence_landmarks":int(spatial.landmark_id.nunique()) if len(spatial) else 0,
        "scientific_hierarchy_changed":False,
        "new_coarse_graining_performed":False,
        "status":"PASS" if lineage_ok else "HOLD"
    }
    (out/"ledger_partition_reconstruction_certificate.json").write_text(json.dumps(cert,indent=2)+"\n")
    return cert

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v1092_ledger_partition_reconstruction.json")
    a=ap.parse_args();project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())

    print("STRATA 1.0.9.2 | Ledger partition reconstruction")
    print("Node-statistic snapshots are not treated as membership tables.")
    print("Level-0 N is derived from the native specimen input.")
    print("Active partitions are recovered only from exact cell-assignment/checkpoint representations.")
    print("A partition must exactly match N0 and the frozen landmark node count.")
    print("Native Xenium H5 expression and cells.parquet coordinates are used for coherence when exact labels are available.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result();reps.append(r)
            print(f"[DONE] {r['sample']}: N0={r['level0_cells']} "
                  f"partitions={r['partitions_resolved']}/{r['landmarks']} "
                  f"nested={r['nested_lineage_all_pairs']} "
                  f"expr={r['expression_coherence_landmarks']}/{r['landmarks']} "
                  f"spatial={r['spatial_coherence_landmarks']}/{r['landmarks']} {r['status']}",flush=True)

    reps.sort(key=lambda x:SAMPLES.index(x["sample"]))
    lineage=all(r["lineage_mass_conservation_certified"] for r in reps)
    expr=all(r["expression_coherence_landmarks"]==r["landmarks"] for r in reps)
    spatial=all(r["spatial_coherence_landmarks"]==r["landmarks"] for r in reps)
    out=project/"results"/"hierarchy_v1092_ledger_partition_reconstruction";out.mkdir(parents=True,exist_ok=True)
    g={
        "strata_version":"1.0.9.2","sample_reports":reps,
        "LEDGER_PARTITION_RECONSTRUCTION_GATE":"PASS" if lineage else "HOLD",
        "LINEAGE_MASS_CONSERVATION_CERTIFIED":bool(lineage),
        "EXPRESSION_COHERENCE_READY":bool(expr),
        "SPATIAL_COHERENCE_READY":bool(spatial),
        "TERMINAL_RULE_PREREQUISITES_READY":bool(lineage and expr and spatial),
        "SCIENTIFIC_HIERARCHY_CHANGED":False,"NEW_COARSE_GRAINING_PERFORMED":False
    }
    p=out/"ledger_partition_reconstruction_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nLEDGER PARTITION RECONSTRUCTION GATE: {g['LEDGER_PARTITION_RECONSTRUCTION_GATE']}")
    print(f"LINEAGE MASS CONSERVATION CERTIFIED: {g['LINEAGE_MASS_CONSERVATION_CERTIFIED']}")
    print(f"EXPRESSION COHERENCE READY: {g['EXPRESSION_COHERENCE_READY']}")
    print(f"SPATIAL COHERENCE READY: {g['SPATIAL_COHERENCE_READY']}")
    print(f"TERMINAL RULE PREREQUISITES READY: {g['TERMINAL_RULE_PREREQUISITES_READY']}")
    print(f"Certificate: {p}")
if __name__=="__main__":main()
