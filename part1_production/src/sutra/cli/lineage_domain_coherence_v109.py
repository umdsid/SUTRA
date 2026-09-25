from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow as pa,pyarrow.parquet as pq

from sutra.hierarchy.v109.audit import *

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def req(p):
    p=Path(p)
    if not p.exists():
        raise FileNotFoundError(p)
    return p

def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def one(project_s,sample,cfg):
    project=Path(project_s)
    v1061=project/"results"/"hierarchy_v1061_full_landmark_materialization"/sample
    table=pd.read_parquet(req(v1061/"landmark_table.parquet"))
    manifest=pd.read_parquet(req(
        project/"results"/"hierarchy_v104_persistence_basin_landmarks"/sample/
        "landmark_spatial_gene_state.parquet"
    ))

    expr_src=find_expression_table(project,sample,cfg)
    coord_src=find_coordinate_table(project,sample,cfg)

    states={}
    partition_rows=[]
    dominant_parts=[]
    expr_parts=[]
    spatial_parts=[]

    for _,lm in table.sort_values("minimum_removed_fraction").iterrows():
        lid=str(lm.landmark_id)
        p=state_path_from_manifest(manifest,int(lm.candidate_landmark))
        state=read_state(p)
        states[lid]=state
        mmap,_=membership_sets(state)
        paudit=partition_audit(mmap)
        partition_rows.append({
            "landmark_id":lid,
            "minimum_nodes":int(lm.minimum_nodes),
            "minimum_removed_fraction":float(lm.minimum_removed_fraction),
            "state_path":str(p) if p else None,
            **paudit
        })

        dom=dominant_by_mass(state,float(cfg["dominant_mass_fraction"]))
        if len(dom):
            dom.insert(0,"landmark_id",lid)
            dominant_parts.append(dom)

        ex=expression_coherence(state,dom,expr_src,cfg)
        if len(ex):
            ex.insert(0,"landmark_id",lid)
            expr_parts.append(ex)

        sp=spatial_coherence(state,dom,coord_src)
        if len(sp):
            sp.insert(0,"landmark_id",lid)
            spatial_parts.append(sp)

    partitions=pd.DataFrame(partition_rows)
    dominant=pd.concat(dominant_parts,ignore_index=True) if dominant_parts else pd.DataFrame()
    expr=pd.concat(expr_parts,ignore_index=True) if expr_parts else pd.DataFrame()
    spatial=pd.concat(spatial_parts,ignore_index=True) if spatial_parts else pd.DataFrame()

    # Consecutive landmark nesting audit.
    ordered=table.sort_values("minimum_removed_fraction").reset_index(drop=True)
    lineage_parts=[];pair_rows=[]
    for i in range(len(ordered)-1):
        child_id=str(ordered.iloc[i].landmark_id)
        parent_id=str(ordered.iloc[i+1].landmark_id)
        cmap,_=membership_sets(states.get(child_id))
        pmap,_=membership_sets(states.get(parent_id))
        d,s=compare_partitions(cmap,pmap)
        if len(d):
            d.insert(0,"child_landmark_id",child_id)
            d.insert(1,"parent_landmark_id",parent_id)
            lineage_parts.append(d)
        pair_rows.append({
            "child_landmark_id":child_id,
            "parent_landmark_id":parent_id,
            **s
        })
    lineage=pd.concat(lineage_parts,ignore_index=True) if lineage_parts else pd.DataFrame()
    pairs=pd.DataFrame(pair_rows)

    # Total mass conservation across frozen landmarks.
    unique_counts=partitions.unique_level0_members.to_numpy(float) if "unique_level0_members" in partitions.columns else np.array([])
    finite_counts=unique_counts[np.isfinite(unique_counts)]
    total_mass_constant=bool(len(finite_counts)==len(partitions) and len(set(finite_counts.astype(int)))==1) if len(partitions) else False
    disjoint_all=bool(partitions.partition_is_disjoint.all()) if len(partitions) else False
    nested_all=bool(pairs.mass_monotone_all.all()) if len(pairs) else False
    no_orphans=bool((pairs.children_orphaned==0).all()) if len(pairs) else False
    no_ambiguous=bool((pairs.children_ambiguous==0).all()) if len(pairs) else False
    lineage_certified=bool(total_mass_constant and disjoint_all and nested_all and no_orphans and no_ambiguous)

    expr_landmarks=int(expr.landmark_id.nunique()) if len(expr) else 0
    spatial_landmarks=int(spatial.landmark_id.nunique()) if len(spatial) else 0
    n_landmarks=int(len(table))

    out=project/"results"/"hierarchy_v109_lineage_conservation_domain_coherence"/sample
    out.mkdir(parents=True,exist_ok=True)
    wpq(partitions,out/"landmark_partition_audit.parquet")
    wpq(pairs,out/"landmark_pair_lineage_audit.parquet")
    wpq(lineage,out/"supernode_lineage_nesting.parquet")
    wpq(dominant,out/"dominant_mass_supernodes.parquet")
    wpq(expr,out/"dominant_expression_coherence.parquet")
    wpq(spatial,out/"dominant_spatial_coherence.parquet")

    cert={
        "sample":sample,
        "landmarks":n_landmarks,
        "total_level0_mass_constant":total_mass_constant,
        "partition_disjoint_all_landmarks":disjoint_all,
        "nested_lineage_all_pairs":nested_all,
        "no_orphan_children":no_orphans,
        "no_ambiguous_parentage":no_ambiguous,
        "lineage_mass_conservation_certified":lineage_certified,
        "dominant_mass_fraction":float(cfg["dominant_mass_fraction"]),
        "dominant_expression_coherence_landmarks":expr_landmarks,
        "dominant_spatial_coherence_landmarks":spatial_landmarks,
        "expression_source_resolved":bool(expr_src is not None),
        "coordinate_source_resolved":bool(coord_src is not None),
        "scientific_hierarchy_changed":False,
        "new_coarse_graining_performed":False,
        "status":"PASS" if lineage_certified else "HOLD"
    }
    (out/"lineage_conservation_domain_coherence_certificate.json").write_text(json.dumps(cert,indent=2)+"\n")
    return cert

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v109_lineage_domain_coherence.json")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())

    c=json.loads(req(
        project/"results"/"hierarchy_v108_supernode_mass_spectrum"/
        "mass_spectrum_global_certificate.json"
    ).read_text())
    if c.get("MASS_HETEROGENEITY_FIGURES_READY") is not True:
        raise SystemExit("ERROR: v1.0.8 mass-spectrum audit not certified")

    print("STRATA 1.0.9 | Lineage conservation + dominant-domain coherence audit")
    print("Frozen hierarchy is unchanged.")
    print("Certifying exact Level-0 partition conservation and monotone lineage mass.")
    print("Dominant domains are the smallest supernode set carrying the configured tissue-mass fraction.")
    print("Expression and spatial coherence use exact Level-0 membership only; unavailable sources remain unresolved.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result();reps.append(r)
            print(
                f"[DONE] {r['sample']}: lineage={r['lineage_mass_conservation_certified']} "
                f"expr={r['dominant_expression_coherence_landmarks']}/{r['landmarks']} "
                f"spatial={r['dominant_spatial_coherence_landmarks']}/{r['landmarks']} "
                f"{r['status']}",flush=True
            )

    reps.sort(key=lambda x:SAMPLES.index(x["sample"]))
    lineage_gate=all(r["lineage_mass_conservation_certified"] for r in reps)
    expr_ready=all(r["dominant_expression_coherence_landmarks"]==r["landmarks"] for r in reps)
    spatial_ready=all(r["dominant_spatial_coherence_landmarks"]==r["landmarks"] for r in reps)

    out=project/"results"/"hierarchy_v109_lineage_conservation_domain_coherence"
    out.mkdir(parents=True,exist_ok=True)
    g={
        "strata_version":"1.0.9",
        "stage":"lineage conservation and dominant-domain coherence audit",
        "sample_reports":reps,
        "LINEAGE_CONSERVATION_GATE":"PASS" if lineage_gate else "HOLD",
        "EXPRESSION_COHERENCE_READY":bool(expr_ready),
        "SPATIAL_COHERENCE_READY":bool(spatial_ready),
        "TERMINAL_RULE_PREREQUISITES_READY":bool(lineage_gate and expr_ready and spatial_ready),
        "SCIENTIFIC_HIERARCHY_CHANGED":False,
        "NEW_COARSE_GRAINING_PERFORMED":False
    }
    p=out/"lineage_conservation_domain_coherence_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nLINEAGE CONSERVATION GATE: {g['LINEAGE_CONSERVATION_GATE']}")
    print(f"EXPRESSION COHERENCE READY: {g['EXPRESSION_COHERENCE_READY']}")
    print(f"SPATIAL COHERENCE READY: {g['SPATIAL_COHERENCE_READY']}")
    print(f"TERMINAL RULE PREREQUISITES READY: {g['TERMINAL_RULE_PREREQUISITES_READY']}")
    print("SCIENTIFIC HIERARCHY CHANGED: False")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
