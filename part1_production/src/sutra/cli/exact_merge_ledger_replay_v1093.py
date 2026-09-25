from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow as pa,pyarrow.parquet as pq

from sutra.hierarchy.v1093.replay import *

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def req(p):
    p=Path(p)
    if not p.exists():raise FileNotFoundError(p)
    return p
def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def derive_n0(project,sample):
    p=project/"data"/sample/"cells.parquet"
    if p.exists():
        return int(pq.ParquetFile(p).metadata.num_rows),str(p)
    raise RuntimeError(f"{sample}: cells.parquet unavailable")

def one(project_s,sample,cfg):
    project=Path(project_s)
    table=pd.read_parquet(req(
        project/"results"/"hierarchy_v1061_full_landmark_materialization"/sample/"landmark_table.parquet"
    )).sort_values("minimum_removed_fraction").reset_index(drop=True)
    target_counts=list(map(int,table.minimum_nodes))
    n0,n0src=derive_n0(project,sample)

    initial,initial_path,initial_key=load_initial_labels(project,sample,n0,cfg)
    if initial is None:
        raise RuntimeError(f"{sample}: no Level-0 label vector found")

    discovery,paths=discover_merge_tables(project,sample,cfg)
    candidate_rows=[];successful=[]
    for p in paths:
        d=read_table(p)
        if d is None:continue
        try:r=replay_merge_table(d,initial,target_counts)
        except Exception as e:
            candidate_rows.append({"path":str(p),"replay_ok":False,"error":repr(e)})
            continue
        resolved=sorted(set(target_counts)&set(r["snapshots"].keys()),reverse=True)
        rec={
            "path":str(p),"replay_ok":True,
            "initial_live":r["initial_live"],"final_live":r["final_live"],
            "applied_merges":r["applied_merges"],
            "unresolved_rows":r["unresolved_rows"],
            "redundant_rows":r["redundant_rows"],
            "landmarks_resolved":int(len(resolved)),
            "landmarks_expected":int(len(target_counts)),
            "all_landmarks_resolved":bool(len(resolved)==len(target_counts)),
            "resolved_counts_json":json.dumps(resolved),
            "schema_json":json.dumps(r["schema"])
        }
        candidate_rows.append(rec)
        if rec["all_landmarks_resolved"]:
            successful.append((p,r))

    replay_audit=pd.DataFrame(candidate_rows)
    # Require one unique replay-capable merger source.
    if len(successful)!=1:
        chosen=None;result=None
    else:
        chosen,result=successful[0]

    out=project/"results"/"hierarchy_v1093_exact_merge_ledger_replay"/sample
    out.mkdir(parents=True,exist_ok=True)
    wpq(discovery,out/"merge_source_discovery.parquet")
    wpq(replay_audit,out/"merge_replay_candidate_audit.parquet")

    snapshot_rows=[];pair_rows=[];lineage_parts=[];expr_parts=[];spatial_parts=[];mass_parts=[]
    labels_by_lid={}
    X,genes,expr_src=load_xenium_expression(project,sample)
    coords,coord_src=load_xenium_coordinates(project,sample,n0)

    if result is not None:
        ldir=out/"landmark_labels";ldir.mkdir(exist_ok=True)
        for _,lm in table.iterrows():
            lid=str(lm.landmark_id);count=int(lm.minimum_nodes)
            lab=result["snapshots"][count]
            labels_by_lid[lid]=lab
            chk=audit_snapshot(lab,n0,count)
            snapshot_rows.append({"landmark_id":lid,"minimum_nodes":count,**chk})
            np.savez_compressed(ldir/f"{lid.replace(':','__')}.npz",labels=lab)

            dom,mass=dominant_labels(lab,float(cfg["dominant_mass_fraction"]))
            mass.insert(0,"landmark_id",lid);mass_parts.append(mass)

            e=expression_coherence(lab,X,dom)
            if len(e):e.insert(0,"landmark_id",lid);expr_parts.append(e)
            s=spatial_coherence(lab,coords,dom)
            if len(s):s.insert(0,"landmark_id",lid);spatial_parts.append(s)

        for i in range(len(table)-1):
            a=str(table.iloc[i].landmark_id);b=str(table.iloc[i+1].landmark_id)
            d,ok=nested_labels(labels_by_lid[a],labels_by_lid[b])
            if len(d):
                d.insert(0,"fine_landmark_id",a);d.insert(1,"coarse_landmark_id",b);lineage_parts.append(d)
            pair_rows.append({"fine_landmark_id":a,"coarse_landmark_id":b,"nested":bool(ok)})

    snapshots=pd.DataFrame(snapshot_rows)
    pairs=pd.DataFrame(pair_rows)
    lineage=pd.concat(lineage_parts,ignore_index=True) if lineage_parts else pd.DataFrame()
    expr=pd.concat(expr_parts,ignore_index=True) if expr_parts else pd.DataFrame()
    spatial=pd.concat(spatial_parts,ignore_index=True) if spatial_parts else pd.DataFrame()
    mass=pd.concat(mass_parts,ignore_index=True) if mass_parts else pd.DataFrame()

    wpq(snapshots,out/"landmark_snapshot_audit.parquet")
    wpq(pairs,out/"landmark_pair_nesting.parquet")
    wpq(lineage,out/"supernode_lineage_nesting.parquet")
    wpq(mass,out/"landmark_mass_spectrum_exact.parquet")
    wpq(expr,out/"dominant_expression_coherence.parquet")
    wpq(spatial,out/"dominant_spatial_coherence.parquet")

    snapshots_ok=bool(len(snapshots)==len(table) and snapshots.cell_count_match.all() and snapshots.node_count_match.all()) if len(snapshots) else False
    nesting_ok=bool(len(pairs)==len(table)-1 and pairs.nested.all()) if len(table)>1 else snapshots_ok
    lineage_ok=bool(snapshots_ok and nesting_ok)
    expr_ready=bool(expr_src is not None and len(expr) and expr.landmark_id.nunique()==len(table))
    spatial_ready=bool(coord_src is not None and len(spatial) and spatial.landmark_id.nunique()==len(table))

    cert={
        "sample":sample,"level0_cells":n0,"level0_source":n0src,
        "initial_label_source":initial_path,"initial_label_key":initial_key,
        "merge_source":str(chosen) if chosen else None,
        "merge_source_unique":bool(len(successful)==1),
        "landmarks":int(len(table)),
        "landmark_partitions_materialized":int(len(snapshots)),
        "all_landmarks_exact":snapshots_ok,
        "nested_lineage_all_pairs":nesting_ok,
        "lineage_mass_conservation_certified":lineage_ok,
        "expression_source":expr_src,
        "coordinate_source":coord_src,
        "expression_coherence_ready":expr_ready,
        "spatial_coherence_ready":spatial_ready,
        "scientific_hierarchy_changed":False,
        "new_coarse_graining_performed":False,
        "status":"PASS" if lineage_ok else "HOLD"
    }
    (out/"exact_merge_ledger_replay_certificate.json").write_text(json.dumps(cert,indent=2)+"\n")
    return cert

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v1093_exact_merge_ledger_replay.json")
    a=ap.parse_args();project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())

    print("STRATA 1.0.9.3 | Exact merge-ledger replay")
    print("No new coarse-graining; the frozen hierarchy is unchanged.")
    print("Reconstructing landmark partitions by deterministic replay from the Level-0 label vector.")
    print("Merge sources are selected by replay validity, not filename alone.")
    print("Every materialized landmark must exactly match N0 and its frozen node count.")
    print("Consecutive landmark partitions must be nested with non-decreasing parent mass.")
    print("Native Xenium expression and coordinates are attached to dominant mass-bearing domains.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result();reps.append(r)
            print(
                f"[DONE] {r['sample']}: partitions={r['landmark_partitions_materialized']}/{r['landmarks']} "
                f"lineage={r['lineage_mass_conservation_certified']} "
                f"expr={r['expression_coherence_ready']} spatial={r['spatial_coherence_ready']} "
                f"{r['status']}",flush=True
            )

    reps.sort(key=lambda x:SAMPLES.index(x["sample"]))
    lineage=all(r["lineage_mass_conservation_certified"] for r in reps)
    expr=all(r["expression_coherence_ready"] for r in reps)
    spatial=all(r["spatial_coherence_ready"] for r in reps)

    out=project/"results"/"hierarchy_v1093_exact_merge_ledger_replay"
    out.mkdir(parents=True,exist_ok=True)
    g={
        "strata_version":"1.0.9.3","stage":"exact merge-ledger replay",
        "sample_reports":reps,
        "EXACT_MERGE_LEDGER_REPLAY_GATE":"PASS" if lineage else "HOLD",
        "LINEAGE_MASS_CONSERVATION_CERTIFIED":bool(lineage),
        "EXPRESSION_COHERENCE_READY":bool(expr),
        "SPATIAL_COHERENCE_READY":bool(spatial),
        "TERMINAL_RULE_PREREQUISITES_READY":bool(lineage and expr and spatial),
        "READY_TO_FREEZE_PRODUCTION_TERMINAL_RULE":bool(lineage and expr and spatial),
        "SCIENTIFIC_HIERARCHY_CHANGED":False,
        "NEW_COARSE_GRAINING_PERFORMED":False
    }
    p=out/"exact_merge_ledger_replay_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nEXACT MERGE LEDGER REPLAY GATE: {g['EXACT_MERGE_LEDGER_REPLAY_GATE']}")
    print(f"LINEAGE MASS CONSERVATION CERTIFIED: {g['LINEAGE_MASS_CONSERVATION_CERTIFIED']}")
    print(f"EXPRESSION COHERENCE READY: {g['EXPRESSION_COHERENCE_READY']}")
    print(f"SPATIAL COHERENCE READY: {g['SPATIAL_COHERENCE_READY']}")
    print(f"TERMINAL RULE PREREQUISITES READY: {g['TERMINAL_RULE_PREREQUISITES_READY']}")
    print(f"READY TO FREEZE PRODUCTION TERMINAL RULE: {g['READY_TO_FREEZE_PRODUCTION_TERMINAL_RULE']}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
