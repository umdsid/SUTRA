
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

def ensure(p:Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def gini(x):
    x=np.asarray(x,dtype=float)
    x=x[np.isfinite(x) & (x>=0)]
    if len(x)==0 or x.sum()==0: return np.nan
    x=np.sort(x)
    n=len(x)
    return float((2*np.sum((np.arange(1,n+1))*x)/(n*x.sum()))-(n+1)/n)

def neff(x):
    x=np.asarray(x,dtype=float)
    s=x.sum()
    if s<=0:return np.nan
    p=x/s
    return float(1.0/np.sum(p*p))

def kfrac(x,f=.8):
    x=np.sort(np.asarray(x,dtype=float))[::-1]
    if len(x)==0 or x.sum()<=0:return np.nan
    return int(np.searchsorted(np.cumsum(x)/x.sum(),f,side="left")+1)

def mass_stats(labels):
    _,m=np.unique(np.asarray(labels),return_counts=True)
    m=m.astype(int)
    return {
        "n_nodes":int(len(m)),
        "total_mass":int(m.sum()),
        "gini":gini(m),
        "n_eff":neff(m),
        "n_eff_fraction":float(neff(m)/len(m)) if len(m) else np.nan,
        "k50":kfrac(m,.5),"k80":kfrac(m,.8),"k90":kfrac(m,.9),
        "max_mass":int(m.max()) if len(m) else 0,
        "max_mass_fraction":float(m.max()/m.sum()) if len(m) else np.nan,
        "masses":m
    }

def normalized_mass_vector(labels):
    _,m=np.unique(np.asarray(labels),return_counts=True)
    m=np.asarray(m,dtype=float)
    if m.sum()==0:return np.array([])
    return np.sort(m/m.sum())

def wasserstein_mass(a,b):
    from scipy.stats import wasserstein_distance
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    if len(a)==0 or len(b)==0:return np.nan
    return float(wasserstein_distance(a,b))

def subsample_labels(labels,frac,rng):
    labels=np.asarray(labels)
    n=len(labels)
    keep=np.sort(rng.choice(n,size=max(2,int(round(frac*n))),replace=False))
    return labels[keep]

def random_coalescent_masses(n0,target,rng):
    # Kingman-like random pair coalescent. Produces a genuine coarsening null,
    # rather than an artificially balanced random partition.
    masses=[1]*int(n0)
    while len(masses)>int(target):
        i,j=sorted(map(int,rng.choice(len(masses),size=2,replace=False)),reverse=True)
        a=masses.pop(i); b=masses.pop(j)
        masses.append(a+b)
    return np.asarray(masses,dtype=int)

def spatial_voronoi_masses(P,target,rng):
    from scipy.spatial import cKDTree
    n=len(P); target=int(target)
    if target>=n:return np.ones(n,dtype=int)
    seed_idx=rng.choice(n,size=target,replace=False)
    tree=cKDTree(np.asarray(P)[seed_idx])
    _,lab=tree.query(np.asarray(P),k=1)
    return np.bincount(lab,minlength=target).astype(int)

def load_landmark_rows(project:Path,sample:str,cfg:dict):
    p=project/cfg["inputs"]["final_landscape"]/sample/"final_supported_pareto_landmarks.csv"
    d=pd.read_csv(p).sort_values("center_eval").reset_index(drop=True)
    base=project/cfg["inputs"]["merger_atlas"]/sample
    out=[]
    for i,r in d.iterrows():
        ce,cn=int(r["center_eval"]),int(r["center_nodes"])
        cp=base/f"labels_center_eval_{ce:03d}_nodes_{cn}.npz"
        if not cp.exists(): raise FileNotFoundError(cp)
        z=np.load(cp,allow_pickle=False)
        key="labels" if "labels" in z.files else z.files[0]
        out.append((i+1,r,np.asarray(z[key]),cp))
    return out

def load_coords(project:Path,sample:str):
    p=project/"data"/sample/"cells.parquet"
    d=pd.read_parquet(p)
    xc=next((c for c in ["x_centroid","x","cell_centroid_x","center_x"] if c in d.columns),None)
    yc=next((c for c in ["y_centroid","y","cell_centroid_y","center_y"] if c in d.columns),None)
    if xc is None or yc is None:
        return None,p
    return d[[xc,yc]].to_numpy(dtype=float),p

def representative_landmarks(rows,n=5):
    if len(rows)<=n:return rows
    idx=np.unique(np.round(np.linspace(0,len(rows)-1,n)).astype(int))
    return [rows[int(i)] for i in idx]

def bootstrap_stability(project:Path,sample:str,cfg:dict,outdir:Path):
    scfg=cfg["stability"]
    rng=np.random.default_rng(int(scfg["random_seed"]) + abs(hash(sample))%100000)
    rows=load_landmark_rows(project,sample,cfg)
    records=[]
    for lid,r,labels,src in rows:
        full=mass_stats(labels)
        fullv=normalized_mass_vector(labels)
        for frac in scfg["cell_retention"]:
            for rep in range(int(scfg["bootstrap_replicates"])):
                q=subsample_labels(labels,float(frac),rng)
                st=mass_stats(q); v=normalized_mass_vector(q)
                records.append({
                    "sample":sample,"landmark_id":lid,"center_nodes":int(r["center_nodes"]),
                    "retention":float(frac),"replicate":rep,
                    "gini":st["gini"],"n_eff_fraction":st["n_eff_fraction"],
                    "max_mass_fraction":st["max_mass_fraction"],
                    "mass_wasserstein":wasserstein_mass(fullv,v)
                })
    d=pd.DataFrame(records)
    ensure(outdir/sample/"stability")
    d.to_csv(outdir/sample/"stability"/"cell_subsampling_stability.csv",index=False)
    return d

def null_model_stability(project:Path,sample:str,cfg:dict,outdir:Path):
    scfg=cfg["stability"]
    rng=np.random.default_rng(int(scfg["random_seed"])+700000+abs(hash(sample))%100000)
    rows=representative_landmarks(load_landmark_rows(project,sample,cfg),int(scfg["representative_landmarks"]))
    P,psrc=load_coords(project,sample)
    records=[]
    for lid,r,labels,src in rows:
        obs=mass_stats(labels); obsv=normalized_mass_vector(labels)
        n0=len(labels); target=int(r["center_nodes"])
        for rep in range(int(scfg["null_replicates"])):
            rm=random_coalescent_masses(n0,target,rng)
            rst={"gini":gini(rm),"n_eff_fraction":neff(rm)/len(rm),"max_mass_fraction":rm.max()/rm.sum()}
            records.append({
                "sample":sample,"landmark_id":lid,"center_nodes":target,"model":"random_coalescent","replicate":rep,
                "gini":rst["gini"],"n_eff_fraction":rst["n_eff_fraction"],
                "max_mass_fraction":rst["max_mass_fraction"],
                "mass_wasserstein":wasserstein_mass(obsv,np.sort(rm/rm.sum()))
            })
            if P is not None:
                sm=spatial_voronoi_masses(P,target,rng)
                records.append({
                    "sample":sample,"landmark_id":lid,"center_nodes":target,"model":"spatial_voronoi","replicate":rep,
                    "gini":gini(sm),"n_eff_fraction":neff(sm)/len(sm),
                    "max_mass_fraction":sm.max()/sm.sum(),
                    "mass_wasserstein":wasserstein_mass(obsv,np.sort(sm/sm.sum()))
                })
        records.append({
            "sample":sample,"landmark_id":lid,"center_nodes":target,"model":"STRATA","replicate":-1,
            "gini":obs["gini"],"n_eff_fraction":obs["n_eff_fraction"],
            "max_mass_fraction":obs["max_mass_fraction"],"mass_wasserstein":0.0
        })
    d=pd.DataFrame(records)
    ensure(outdir/sample/"stability")
    d.to_csv(outdir/sample/"stability"/"reduction_scheme_nulls.csv",index=False)
    return d

def gene_tables(project:Path,sample:str,cfg:dict):
    base=project/cfg["inputs"]["brain_pipeline"]/sample/"merger_biology"
    return sorted(base.glob("landmark_*_gene_merger_scores.csv"))

def build_gene_trajectory(project:Path,sample:str,cfg:dict,outdir:Path):
    tabs=gene_tables(project,sample,cfg)
    if not tabs:return pd.DataFrame()
    rows=[]
    for p in tabs:
        d=pd.read_csv(p)
        lid=int(d["landmark_id"].iloc[0]) if "landmark_id" in d.columns else int(p.stem.split("_")[1])
        for _,r in d.iterrows():
            if bool(r.get("gene_passes_detection_filter",True)):
                rows.append({"sample":sample,"landmark_id":lid,"gene":r["gene"],
                             "score":float(r["merger_concordance_score"]),
                             "z":float(r["merger_concordance_z"])})
    A=pd.DataFrame(rows)
    if len(A):
        ensure(outdir/sample/"gene_trajectories")
        A.to_csv(outdir/sample/"gene_trajectories"/"gene_landmark_trajectories.csv",index=False)
    return A

def read_support_trajectory(project:Path,sample:str,cfg:dict,outdir:Path):
    base=project/cfg["inputs"]["production"]/sample
    hits=sorted(base.rglob("scientific_evaluations*.parquet"))
    if not hits:return pd.DataFrame()
    d=pd.read_parquet(hits[0]).copy()
    low={str(c).lower():c for c in d.columns}
    def resolve(tokens):
        for c in d.columns:
            s=str(c).lower()
            if all(t in s for t in tokens):return c
        return None
    evalc=resolve(["eval"]) or resolve(["evaluation"])
    nodec=resolve(["node"]) or resolve(["nodes"])
    keep=[]
    for kind in ["mass","expr","spatial","slow"]:
        c=next((col for col in d.columns if kind in str(col).lower()),None)
        if c is not None:keep.append(c)
    cols=[c for c in [evalc,nodec]+keep if c is not None]
    q=d[cols].copy() if cols else pd.DataFrame()
    if len(q):
        ensure(outdir/sample/"support")
        q.to_csv(outdir/sample/"support"/"scientific_support_trajectory.csv",index=False)
    return q
