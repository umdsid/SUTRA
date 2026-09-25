from __future__ import annotations
import argparse,json,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from scipy.stats import spearmanr
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import r2_score, mean_absolute_error
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

from strata_native_mechanics.patches import partition_core_cells,add_halo
from strata_native_mechanics.nullspace_forensics import patch_features

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def load_cache(project,s):
    c=project/"results"/"native_mechanics_cache"/s
    E=pd.read_parquet(c/"interfaces.parquet")
    B=pd.read_parquet(c/"background_components.parquet")
    J=pd.read_json(c/"junctions.jsonl",lines=True)
    C=pd.read_parquet(c/"cells.parquet")
    if "incident_interfaces" in J.columns:
        J["incident_interfaces"]=J["incident_interfaces"].apply(
            lambda x:list(x) if isinstance(x,(list,tuple,np.ndarray)) else [])
    return E,B,J,C

def one_sample(project_s,s,patch_size,halo):
    project=Path(project_s)
    E,B,J,C=load_cache(project,s)
    ident=pd.read_csv(project/"results"/"native_identifiability_v062"/s/"patch_identifiability.csv")
    cores=partition_core_cells(E,patch_size=patch_size)
    patches=[add_halo(c,E,hops=halo) for c in cores]
    rows=[]
    for pid,cells in enumerate(patches):
        f=patch_features(cells,E,B,J,C)
        f.update({"sample":s,"patch_id":pid})
        rows.append(f)
    feat=pd.DataFrame(rows)
    out=feat.merge(ident,on=["sample","patch_id","n_patch_cells"],how="left",suffixes=("","_ident"))
    return s,out

def residualize(y,X):
    X=np.asarray(X,float); y=np.asarray(y,float)
    X=np.c_[np.ones(len(X)),X]
    beta,*_=np.linalg.lstsq(X,y,rcond=None)
    return y-X@beta

def association_table(df,features):
    rows=[]
    y=df["structural_nullity"].to_numpy(float)
    size=np.c_[df["n_patch_cells"].to_numpy(float),
               df["n_variables"].to_numpy(float)]
    yr=residualize(y,size)
    for c in features:
        x=pd.to_numeric(df[c],errors="coerce").to_numpy(float)
        q=np.isfinite(x)&np.isfinite(y)
        if q.sum()<20: continue
        rho,p=spearmanr(x[q],y[q])
        xr=residualize(x[q],size[q])
        rho_adj,p_adj=spearmanr(xr,yr[q])
        rows.append({
            "feature":c,
            "spearman_rho_nullity":float(rho),
            "spearman_p":float(p),
            "size_adjusted_spearman_rho":float(rho_adj),
            "size_adjusted_p":float(p_adj),
            "n":int(q.sum()),
        })
    return pd.DataFrame(rows)

def fit_model(df,features):
    if not HAVE_SKLEARN:
        return (
            pd.DataFrame([{
                "fold": -1,
                "train_samples": [],
                "test_samples": [],
                "r2_log_nullity": np.nan,
                "mae_log_nullity": np.nan,
                "n_test": 0,
                "status": "SKIP_NO_SKLEARN",
            }]),
            pd.DataFrame({
                "feature": features,
                "permutation_importance_mean": np.nan,
                "permutation_importance_sd": np.nan,
                "mean_within_fold_sd": np.nan,
            })
        )
    X=df[features].apply(pd.to_numeric,errors="coerce")
    X=X.replace([np.inf,-np.inf],np.nan)
    med=X.median()
    X=X.fillna(med)
    y=np.log1p(df["structural_nullity"].to_numpy(float))
    groups=df["sample"].astype(str).to_numpy()

    # Hold out one full sample to prevent spatial/within-sample leakage.
    splitter=GroupShuffleSplit(n_splits=3,test_size=1/3,random_state=17)
    fold_rows=[]; importances=[]
    for fold,(tr,te) in enumerate(splitter.split(X,y,groups=groups)):
        rf=RandomForestRegressor(
            n_estimators=350,max_depth=12,min_samples_leaf=4,
            max_features="sqrt",n_jobs=-1,random_state=100+fold
        )
        rf.fit(X.iloc[tr],y[tr])
        pred=rf.predict(X.iloc[te])
        fold_rows.append({
            "fold":fold,
            "train_samples":sorted(set(groups[tr])),
            "test_samples":sorted(set(groups[te])),
            "r2_log_nullity":float(r2_score(y[te],pred)),
            "mae_log_nullity":float(mean_absolute_error(y[te],pred)),
            "n_test":int(len(te)),
        })
        pi=permutation_importance(
            rf,X.iloc[te],y[te],n_repeats=8,random_state=200+fold,n_jobs=-1
        )
        importances.append(pd.DataFrame({
            "feature":features,
            "importance_mean":pi.importances_mean,
            "importance_sd":pi.importances_std,
            "fold":fold
        }))

    imp=pd.concat(importances,ignore_index=True)
    agg=imp.groupby("feature",as_index=False).agg(
        permutation_importance_mean=("importance_mean","mean"),
        permutation_importance_sd=("importance_mean","std"),
        mean_within_fold_sd=("importance_sd","mean")
    ).sort_values("permutation_importance_mean",ascending=False)
    return pd.DataFrame(fold_rows),agg

def suspect_summary(assoc,imp):
    groups={
        "junction_information":[
            "n_junctions","junctions_per_cell","junction_incidence_mean",
            "high_order_junction_fraction"
        ],
        "curvature_information":[
            "curvature_abs_mean","curvature_abs_median","curvature_abs_q90",
            "curvature_confidence_mean","low_information_curvature_fraction"
        ],
        "boundary_exposure":[
            "cell_boundary_interface_fraction","internal_gap_interface_fraction",
            "n_background_components_touched","n_internal_gaps_touched"
        ],
        "patch_topology":[
            "graph_mean_degree","graph_degree_cv","graph_leaf_fraction",
            "graph_n_components","graph_diameter","graph_cycle_rank",
            "graph_mean_clustering"
        ],
        "cell_shape":[
            "cell_area_cv","cell_compactness_mean","cell_compactness_cv",
            "cell_perimeter_sqrt_area_mean","interface_length_cv",
            "interface_orientation_entropy"
        ],
        "patch_size_complexity":[
            "n_patch_cells","n_interfaces","graph_n_edges"
        ],
    }
    amap=assoc.set_index("feature").to_dict("index") if len(assoc) else {}
    imap=imp.set_index("feature").to_dict("index") if len(imp) else {}
    rows=[]
    for g,features in groups.items():
        vals=[]
        for f in features:
            rho=abs(amap.get(f,{}).get("size_adjusted_spearman_rho",np.nan))
            pi=imap.get(f,{}).get("permutation_importance_mean",np.nan)
            vals.append((f,rho,pi))
        finite=[x for x in vals if np.isfinite(x[1]) or np.isfinite(x[2])]
        if finite:
            best=max(finite,key=lambda x:(
                -np.inf if not np.isfinite(x[2]) else x[2],
                -np.inf if not np.isfinite(x[1]) else x[1]
            ))
            rows.append({
                "suspect":g,
                "best_feature":best[0],
                "best_abs_size_adjusted_rho":None if not np.isfinite(best[1]) else float(best[1]),
                "best_permutation_importance":None if not np.isfinite(best[2]) else float(best[2]),
            })
    return pd.DataFrame(rows).sort_values(
        ["best_permutation_importance","best_abs_size_adjusted_rho"],
        ascending=False,na_position="last"
    )

def render_extremes(project,df,n=12):
    try:
        import matplotlib.pyplot as plt
        import tifffile
    except Exception as e:
        return {"status":"SKIP","reason":str(e)}

    out=project/"results"/"native_nullspace_forensics_v063"/"extreme_patch_panels"
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for s in SAMPLES:
        E,B,J,C=load_cache(project,s)
        cores=partition_core_cells(E,patch_size=300)
        patches=[add_halo(c,E,hops=1) for c in cores]
        sdf=df[df["sample"]==s]
        picks=pd.concat([
            sdf.nsmallest(n,"structural_nullity"),
            sdf.nlargest(n,"structural_nullity")
        ]).drop_duplicates("patch_id")
        mask_path=project/"results"/"production_geometry_r5"/s/"production_segmentation.tif"
        mask=tifffile.imread(mask_path)
        for r in picks.itertuples():
            cells=set(patches[int(r.patch_id)])
            ys=[];xs=[]
            for c in cells:
                sl=np.argwhere(mask==c)
                if len(sl):
                    ys.extend([sl[:,0].min(),sl[:,0].max()])
                    xs.extend([sl[:,1].min(),sl[:,1].max()])
            if not ys: continue
            y0=max(0,min(ys)-8);y1=min(mask.shape[0],max(ys)+9)
            x0=max(0,min(xs)-8);x1=min(mask.shape[1],max(xs)+9)
            crop=mask[y0:y1,x0:x1]
            view=np.isin(crop,list(cells))
            fig,ax=plt.subplots(figsize=(5,5))
            ax.imshow(view,interpolation="nearest")
            ax.set_axis_off()
            ax.text(.02,.98,
                    f"{s} patch {r.patch_id}\nnullity={r.structural_nullity}, rank={r.structural_rank_fraction:.3f}",
                    transform=ax.transAxes,va="top",ha="left",
                    bbox=dict(facecolor="white",alpha=.8,edgecolor="none"))
            p=out/f"{s}_patch_{int(r.patch_id):04d}_nullity_{int(r.structural_nullity)}.png"
            fig.savefig(p,dpi=140,bbox_inches="tight");plt.close(fig)
            records.append(str(p))
    return {"status":"PASS","n_images":len(records)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--patch-size",type=int,default=300)
    ap.add_argument("--halo",type=int,default=1)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    print("Building patch forensic descriptors...",flush=True)
    dfs=[]
    with ProcessPoolExecutor(max_workers=min(3,a.workers)) as ex:
        futs={ex.submit(one_sample,str(project),s,a.patch_size,a.halo):s for s in SAMPLES}
        for f in as_completed(futs):
            s,df=f.result();dfs.append(df)
            print(f"  [DONE] {s}: {len(df)} patches",flush=True)
    df=pd.concat(dfs,ignore_index=True)

    reserved={
        "sample","patch_id","structural_nullity","structural_rank",
        "structural_rank_fraction","structurally_identifiable",
        "n_rows","n_variables","young_laplace_rows","junction_x_rows","junction_y_rows"
    }
    features=[
        c for c in df.columns
        if c not in reserved
        and pd.api.types.is_numeric_dtype(df[c])
        and not c.endswith("_nullity")
        and not c.endswith("_rank_fraction")
        and not c.endswith("_nullity_reduction")
        and not c.endswith("_nullity_removed")
        and not c.endswith("_nvars")
    ]

    out=project/"results"/"native_nullspace_forensics_v063"
    out.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out/"patch_forensic_features.parquet",index=False)

    assoc=association_table(df,features)
    assoc.sort_values("size_adjusted_spearman_rho",
                      key=lambda s:s.abs(),ascending=False).to_csv(
        out/"feature_associations.csv",index=False
    )

    folds,imp=fit_model(df,features)
    folds.to_csv(out/"model_validation.csv",index=False)
    imp.to_csv(out/"permutation_importance.csv",index=False)

    suspects=suspect_summary(assoc,imp)
    suspects.to_csv(out/"suspect_ranking.csv",index=False)

    # Basic best/worst feature contrast.
    qlo=df.structural_nullity.quantile(.1)
    qhi=df.structural_nullity.quantile(.9)
    contrast=[]
    for c in features:
        xlo=pd.to_numeric(df.loc[df.structural_nullity<=qlo,c],errors="coerce")
        xhi=pd.to_numeric(df.loc[df.structural_nullity>=qhi,c],errors="coerce")
        if xlo.notna().sum() and xhi.notna().sum():
            contrast.append({
                "feature":c,
                "low_nullity_median":float(xlo.median()),
                "high_nullity_median":float(xhi.median()),
                "median_difference":float(xhi.median()-xlo.median())
            })
    pd.DataFrame(contrast).to_csv(out/"best_worst_patch_contrast.csv",index=False)

    render=render_extremes(project,df,n=8)

    summary={
        "strata_version":"0.6.3.1",
        "sklearn_available": bool(HAVE_SKLEARN),
        "stage":"nullspace forensic feature audit",
        "diagnostic_only":True,
        "n_patches":int(len(df)),
        "n_features":int(len(features)),
        "model_validation":folds.to_dict("records"),
        "top_features":imp.head(15).to_dict("records"),
        "suspect_ranking":suspects.to_dict("records"),
        "render_extremes":render,
        "interpretation_rule":
            "Feature ranking is considered credible only when held-out-sample R2 is meaningfully positive."
    }
    (out/"nullspace_forensics_certificate.json").write_text(json.dumps(summary,indent=2))
    print("\nHeld-out-sample model:")
    print(folds.to_string(index=False))
    print("\nTop permutation features:")
    print(imp.head(12).to_string(index=False))
    print("\nSuspect ranking:")
    print(suspects.to_string(index=False))
    print(f"\nCertificate: {out/'nullspace_forensics_certificate.json'}")

if __name__=="__main__":
    main()
