from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow as pa,pyarrow.parquet as pq
from strata_hierarchy.v10931.replay_shards import *

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")
def req(p):
    p=Path(p)
    if not p.exists():raise FileNotFoundError(p)
    return p
def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")
def n0(project,s):
    p=project/"data"/s/"cells.parquet"
    return int(pq.ParquetFile(p).metadata.num_rows)

def one(project_s,sample,cfg):
    project=Path(project_s)
    table=pd.read_parquet(req(
      project/"results"/"hierarchy_v1061_full_landmark_materialization"/sample/"landmark_table.parquet"
    )).sort_values("minimum_removed_fraction").reset_index(drop=True)
    targets=list(map(int,table.minimum_nodes));N=n0(project,sample)
    initial,ipath,ikey=load_initial_labels(project,sample,N,cfg)
    files,discovery=discover_stream(project,sample,cfg)
    out=project/"results"/"hierarchy_v10931_ordered_merge_shard_replay"/sample
    out.mkdir(parents=True,exist_ok=True)
    wpq(discovery,out/"merge_shard_discovery.parquet")

    if initial is None or files is None or len(files)==0:
        snaps={};shards=pd.DataFrame();stream_ok=False
    else:
        snaps,shards,stream_ok=replay_stream(files,initial,targets)
    wpq(shards,out/"ordered_merge_shard_replay.parquet")

    snaprows=[];labels_by={}
    ldir=out/"landmark_labels";ldir.mkdir(exist_ok=True)
    for _,lm in table.iterrows():
        lid=str(lm.landmark_id);nn=int(lm.minimum_nodes)
        if nn not in snaps:
            snaprows.append({"landmark_id":lid,"expected_nodes":nn,"resolved":False})
            continue
        lab=snaps[nn];a=snapshot_audit(lab,N,nn)
        ok=a["cell_count_match"] and a["node_count_match"]
        snaprows.append({"landmark_id":lid,"expected_nodes":nn,"resolved":ok,**a})
        if ok:
            labels_by[lid]=lab
            np.savez_compressed(ldir/f"{lid.replace(':','__')}.npz",labels=lab)
    snapdf=pd.DataFrame(snaprows);wpq(snapdf,out/"landmark_snapshot_audit.parquet")

    pairs=[];lp=[]
    for i in range(len(table)-1):
        a=str(table.iloc[i].landmark_id);b=str(table.iloc[i+1].landmark_id)
        if a not in labels_by or b not in labels_by:
            pairs.append({"fine_landmark_id":a,"coarse_landmark_id":b,"resolved":False,"nested":False})
            continue
        d,ok=nested_labels(labels_by[a],labels_by[b])
        d.insert(0,"fine_landmark_id",a);d.insert(1,"coarse_landmark_id",b);lp.append(d)
        pairs.append({"fine_landmark_id":a,"coarse_landmark_id":b,"resolved":True,"nested":ok})
    pairdf=pd.DataFrame(pairs);lineage=pd.concat(lp,ignore_index=True) if lp else pd.DataFrame()
    wpq(pairdf,out/"landmark_pair_nesting.parquet");wpq(lineage,out/"supernode_lineage_nesting.parquet")

    X,xsrc=load_native_expression_aligned(project,sample)
    P,psrc=load_coordinates(project,sample,N)
    masses=[];expr=[];sp=[]
    for lid,lab in labels_by.items():
        keep,m=dominant(lab,float(cfg["dominant_mass_fraction"]))
        m.insert(0,"landmark_id",lid);masses.append(m)
        e=expression_coherence(lab,X,keep)
        if len(e):e.insert(0,"landmark_id",lid);expr.append(e)
        q=spatial_coherence(lab,P,keep)
        if len(q):q.insert(0,"landmark_id",lid);sp.append(q)
    mass=pd.concat(masses,ignore_index=True) if masses else pd.DataFrame()
    ex=pd.concat(expr,ignore_index=True) if expr else pd.DataFrame()
    spa=pd.concat(sp,ignore_index=True) if sp else pd.DataFrame()
    wpq(mass,out/"exact_mass_spectrum.parquet");wpq(ex,out/"dominant_expression_coherence.parquet");wpq(spa,out/"dominant_spatial_coherence.parquet")

    all_snap=bool(len(snapdf)==len(table) and snapdf.resolved.fillna(False).all())
    nested=bool(len(pairdf)==len(table)-1 and pairdf.resolved.all() and pairdf.nested.all()) if len(table)>1 else all_snap
    lineage_ok=bool(stream_ok and all_snap and nested)
    expr_ok=bool(xsrc and len(ex) and ex.landmark_id.nunique()==len(table))
    spatial_ok=bool(psrc and len(spa) and spa.landmark_id.nunique()==len(table))
    cert={
      "sample":sample,"level0_cells":N,
      "initial_label_source":ipath,"initial_label_key":ikey,
      "merge_stream_directory":str((project/cfg["v093_ledger_root"]/sample/"merges")),
      "merge_shards_discovered":int(len(files) if files else 0),
      "stream_replay_valid":bool(stream_ok),
      "landmarks":int(len(table)),"landmarks_resolved":int(snapdf.resolved.fillna(False).sum()),
      "all_landmarks_exact":all_snap,"nested_lineage_all_pairs":nested,
      "lineage_mass_conservation_certified":lineage_ok,
      "expression_source":xsrc,"coordinate_source":psrc,
      "expression_coherence_ready":expr_ok,"spatial_coherence_ready":spatial_ok,
      "scientific_hierarchy_changed":False,"new_coarse_graining_performed":False,
      "status":"PASS" if lineage_ok else "HOLD"
    }
    (out/"ordered_merge_shard_replay_certificate.json").write_text(json.dumps(cert,indent=2)+"\n")
    return cert

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v10931_ordered_merge_shard_replay.json")
    a=ap.parse_args();project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())
    print("STRATA 1.0.9.3.1 | Ordered merge-shard replay")
    print("Frozen hierarchy unchanged; no new reduction.")
    print("v0.9.3 merge_XXXXXX parquet files are replayed as one ordered microstep stream.")
    print("Pair schema includes the observed super_i / super_j fields.")
    print("Every batch must be disjoint in the active partition.")
    print("Landmarks are materialized only at actual post-batch node counts; skipped counts are never synthesized.")
    print("Native expression is aligned to cells.parquet by explicit Xenium cell IDs/barcodes.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")
    reps=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result();reps.append(r)
            print(f"[DONE] {r['sample']}: shards={r['merge_shards_discovered']} "
                  f"landmarks={r['landmarks_resolved']}/{r['landmarks']} "
                  f"lineage={r['lineage_mass_conservation_certified']} "
                  f"expr={r['expression_coherence_ready']} spatial={r['spatial_coherence_ready']} {r['status']}",flush=True)
    reps.sort(key=lambda x:SAMPLES.index(x["sample"]))
    lin=all(r["lineage_mass_conservation_certified"] for r in reps)
    ex=all(r["expression_coherence_ready"] for r in reps)
    sp=all(r["spatial_coherence_ready"] for r in reps)
    out=project/"results"/"hierarchy_v10931_ordered_merge_shard_replay";out.mkdir(parents=True,exist_ok=True)
    g={
      "strata_version":"1.0.9.3.1","sample_reports":reps,
      "ORDERED_MERGE_SHARD_REPLAY_GATE":"PASS" if lin else "HOLD",
      "LINEAGE_MASS_CONSERVATION_CERTIFIED":lin,
      "EXPRESSION_COHERENCE_READY":ex,"SPATIAL_COHERENCE_READY":sp,
      "TERMINAL_RULE_PREREQUISITES_READY":bool(lin and ex and sp),
      "READY_TO_FREEZE_PRODUCTION_TERMINAL_RULE":bool(lin and ex and sp),
      "SCIENTIFIC_HIERARCHY_CHANGED":False,"NEW_COARSE_GRAINING_PERFORMED":False
    }
    p=out/"ordered_merge_shard_replay_global_certificate.json";p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nORDERED MERGE SHARD REPLAY GATE: {g['ORDERED_MERGE_SHARD_REPLAY_GATE']}")
    print(f"LINEAGE MASS CONSERVATION CERTIFIED: {g['LINEAGE_MASS_CONSERVATION_CERTIFIED']}")
    print(f"EXPRESSION COHERENCE READY: {g['EXPRESSION_COHERENCE_READY']}")
    print(f"SPATIAL COHERENCE READY: {g['SPATIAL_COHERENCE_READY']}")
    print(f"TERMINAL RULE PREREQUISITES READY: {g['TERMINAL_RULE_PREREQUISITES_READY']}")
    print(f"READY TO FREEZE PRODUCTION TERMINAL RULE: {g['READY_TO_FREEZE_PRODUCTION_TERMINAL_RULE']}")
    print(f"Certificate: {p}")
if __name__=="__main__":main()
