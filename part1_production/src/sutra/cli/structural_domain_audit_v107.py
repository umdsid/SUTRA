from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow as pa,pyarrow.parquet as pq
from sutra.hierarchy.v107.domain_audit import *

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def req(p):
    p=Path(p)
    if not p.exists():raise FileNotFoundError(p)
    return p
def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def one(project_s,sample,cfg):
    project=Path(project_s)
    v1061=project/"results"/"hierarchy_v1061_full_landmark_materialization"/sample
    table=pd.read_parquet(req(v1061/"landmark_table.parquet"))
    lineage=pd.read_parquet(req(v1061/"landmark_lineage.parquet"))
    state_manifest=pd.read_parquet(req(
        project/"results"/"hierarchy_v104_persistence_basin_landmarks"/sample/
        "landmark_spatial_gene_state.parquet"
    ))

    expr_source=find_level0_expression(project,sample,cfg)

    lm_rows=[];unit_parts=[];expr_parts=[];spatial_parts=[];boundary_parts=[]
    for _,lm in table.iterrows():
        lid=str(lm.landmark_id);cand=int(lm.candidate_landmark)
        q=state_manifest[pd.to_numeric(state_manifest.candidate_landmark,errors="coerce")==cand]
        path=None
        if len(q):
            r=q.iloc[0]
            path=r.get("node_state_path",r.get("node_state_source",None))
        state=None
        if path and Path(str(path)).exists():
            try:state=pd.read_parquet(Path(str(path)))
            except Exception:state=None

        # Fallback to explicit lineage mass if state lacks membership.
        mass_col,mass=infer_mass_column(state) if state is not None else (None,None)
        if mass is None and len(lineage):
            lq=lineage[lineage.landmark_id.astype(str)==lid]
            if len(lq) and "level0_members" in lq.columns:
                mass=np.array([len(parse_members(x)) for x in lq.level0_members],float)
                mass_col="level0_members_lineage"
                state=lq.rename(columns={"level0_members":"level0_members"}).copy()

        conc=concentration_summary(mass) if mass is not None else {}

        eh=expression_heterogeneity_from_members(state,expr_source,cfg)
        if len(eh):
            eh.insert(0,"landmark_id",lid);expr_parts.append(eh)

        sp=spatial_compactness(state) if state is not None else pd.DataFrame()
        if len(sp):
            sp.insert(0,"landmark_id",lid);spatial_parts.append(sp)
        bp=boundary_proxy(state) if state is not None else pd.DataFrame()
        if len(bp):
            bp.insert(0,"landmark_id",lid);boundary_parts.append(bp)

        # Per-unit table with mass and expression heterogeneity if alignable.
        units=pd.DataFrame()
        if state is not None and mass is not None:
            idc=next((c for c in ["supernode_id","node_id"] if c in state.columns),None)
            units=pd.DataFrame({
                "landmark_id":lid,
                "row_index":np.arange(len(mass),dtype=int),
                "supernode_id":state[idc].values if idc and len(state)==len(mass) else np.arange(len(mass)),
                "mass":mass
            })
            if len(eh) and "supernode_id" in eh.columns:
                units=units.merge(eh[["supernode_id","within_expression_mse",
                    "within_expression_feature_median_variance","members_with_expression"]],
                    on="supernode_id",how="left")
            if "within_expression_mse" in units.columns:
                labels,mq,hq=classify_units(
                    units.mass.to_numpy(float),
                    units.within_expression_mse.to_numpy(float),cfg
                )
                units["unit_class"]=labels
            else:
                mq=hq=np.nan
            unit_parts.append(units)
        else:
            mq=hq=np.nan

        rec={
            "sample":sample,"landmark_id":lid,
            "minimum_nodes":int(lm.minimum_nodes),
            "minimum_removed_fraction":float(lm.minimum_removed_fraction),
            "mass_source":mass_col,
            "mass_resolved":bool(mass is not None),
            "expression_heterogeneity_resolved":bool(len(eh)>0),
            "spatial_compactness_resolved":bool(len(sp)>0 and len(sp.columns)>2),
            "boundary_proxy_resolved":bool(len(bp)>0 and len(bp.columns)>2),
            "large_mass_quantile_cutoff":float(mq) if np.isfinite(mq) else None,
            "high_heterogeneity_quantile_cutoff":float(hq) if np.isfinite(hq) else None,
            **conc
        }
        if len(eh):
            rec["expression_heterogeneity_median"]=safe_median(eh.within_expression_mse)
            rec["expression_heterogeneity_q90"]=safe_quantile(eh.within_expression_mse,.90)
        lm_rows.append(rec)

    landmarks=pd.DataFrame(lm_rows)
    units=pd.concat(unit_parts,ignore_index=True) if unit_parts else pd.DataFrame()
    expr=pd.concat(expr_parts,ignore_index=True) if expr_parts else pd.DataFrame()
    spatial=pd.concat(spatial_parts,ignore_index=True) if spatial_parts else pd.DataFrame()
    boundary=pd.concat(boundary_parts,ignore_index=True) if boundary_parts else pd.DataFrame()

    out=project/"results"/"hierarchy_v107_structural_domain_audit"/sample
    out.mkdir(parents=True,exist_ok=True)
    wpq(landmarks,out/"landmark_domain_summary.parquet")
    wpq(units,out/"supernode_mass_heterogeneity.parquet")
    wpq(expr,out/"supernode_expression_heterogeneity.parquet")
    wpq(spatial,out/"supernode_spatial_compactness.parquet")
    wpq(boundary,out/"supernode_boundary_proxy.parquet")

    # Mass spectrum ranked within each landmark
    if len(units):
        specs=[]
        for lid,g in units.groupby("landmark_id"):
            z=g.sort_values("mass",ascending=False).copy().reset_index(drop=True)
            z["mass_rank"]=np.arange(1,len(z)+1)
            z["mass_fraction"]=z.mass/z.mass.sum()
            z["cumulative_mass_fraction"]=z.mass_fraction.cumsum()
            specs.append(z[["landmark_id","supernode_id","mass_rank","mass","mass_fraction","cumulative_mass_fraction"]])
        spectrum=pd.concat(specs,ignore_index=True)
    else:spectrum=pd.DataFrame()
    wpq(spectrum,out/"supernode_mass_spectrum.parquet")

    mass_complete=bool(len(landmarks) and landmarks.mass_resolved.all())
    # Expression heterogeneity is a diagnostic availability flag, not a gate,
    # because the exact Level-0 expression table may not be stored in result roots.
    cert={
        "sample":sample,
        "principal_landmarks":int(len(landmarks)),
        "mass_spectrum_resolved_all_landmarks":mass_complete,
        "expression_heterogeneity_resolved_landmarks":int(landmarks.expression_heterogeneity_resolved.sum()),
        "spatial_compactness_resolved_landmarks":int(landmarks.spatial_compactness_resolved.sum()),
        "boundary_proxy_resolved_landmarks":int(landmarks.boundary_proxy_resolved.sum()),
        "effective_domain_numbers":{
            str(r.landmark_id):float(r.effective_domain_number)
            for r in landmarks.itertuples(index=False)
            if hasattr(r,"effective_domain_number") and np.isfinite(r.effective_domain_number)
        },
        "K80":{
            str(r.landmark_id):int(r.K80)
            for r in landmarks.itertuples(index=False)
            if hasattr(r,"K80") and pd.notna(r.K80)
        },
        "formal_supernode_counts":{
            str(r.landmark_id):int(r.formal_supernodes)
            for r in landmarks.itertuples(index=False)
            if hasattr(r,"formal_supernodes") and pd.notna(r.formal_supernodes)
        },
        "scientific_hierarchy_changed":False,
        "new_coarse_graining_performed":False,
        "status":"PASS" if mass_complete else "HOLD"
    }
    (out/"structural_domain_certificate.json").write_text(json.dumps(cert,indent=2)+"\n")
    return cert

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v107_structural_domain_audit.json")
    a=ap.parse_args();project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())
    c=json.loads(req(
        project/"results"/"hierarchy_v1061_full_landmark_materialization"/
        "full_landmark_materialization_global_certificate.json"
    ).read_text())
    if c.get("FULL_LANDMARK_STATE_MATERIALIZED") is not True:
        raise SystemExit("ERROR: v1.0.6.1 full landmark materialization not certified")

    print("STRATA 1.0.7 | Structural domain audit")
    print("Frozen landmark hierarchy is unchanged.")
    print("Auditing formal supernode count versus mass-weighted effective domain count.")
    print("Computing K50/K80/K90, mass Gini, inverse-participation effective domain number.")
    print("Expression heterogeneity is computed only from explicit Level-0 membership + expression tables.")
    print("Spatial compactness/boundary status use existing geometry fields only.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result();reps.append(r)
            print(f"[DONE] {r['sample']}: landmarks={r['principal_landmarks']} "
                  f"mass_all={r['mass_spectrum_resolved_all_landmarks']} "
                  f"expr_het={r['expression_heterogeneity_resolved_landmarks']} "
                  f"spatial={r['spatial_compactness_resolved_landmarks']} "
                  f"boundary={r['boundary_proxy_resolved_landmarks']} {r['status']}",flush=True)
            print(f"       N_eff={r['effective_domain_numbers']}")
            print(f"       K80={r['K80']}")

    reps.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reps)
    out=project/"results"/"hierarchy_v107_structural_domain_audit"
    out.mkdir(parents=True,exist_ok=True)
    g={
        "strata_version":"1.0.7","stage":"structural domain audit",
        "sample_reports":reps,
        "STRUCTURAL_DOMAIN_AUDIT_GATE":"PASS" if gate else "HOLD",
        "MASS_WEIGHTED_DOMAIN_STRUCTURE_READY":bool(gate),
        "SCIENTIFIC_HIERARCHY_CHANGED":False,
        "NEW_COARSE_GRAINING_PERFORMED":False,
        "READY_TO_DECIDE_IF_DEEPER_REDUCTION_IS_NEEDED":bool(gate)
    }
    p=out/"structural_domain_audit_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nSTRUCTURAL DOMAIN AUDIT GATE: {g['STRUCTURAL_DOMAIN_AUDIT_GATE']}")
    print(f"MASS-WEIGHTED DOMAIN STRUCTURE READY: {g['MASS_WEIGHTED_DOMAIN_STRUCTURE_READY']}")
    print("SCIENTIFIC HIERARCHY CHANGED: False")
    print(f"READY TO DECIDE IF DEEPER REDUCTION IS NEEDED: {g['READY_TO_DECIDE_IF_DEEPER_REDUCTION_IS_NEEDED']}")
    print(f"Certificate: {p}")
if __name__=="__main__":main()
