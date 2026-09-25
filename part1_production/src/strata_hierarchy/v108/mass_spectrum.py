from __future__ import annotations
import numpy as np
import pandas as pd

def finite_positive(x):
    a=np.asarray(x,float)
    return a[np.isfinite(a)&(a>0)]

def normalized_mass(mass):
    a=np.asarray(mass,float)
    q=np.isfinite(a)&(a>0)
    out=np.full(len(a),np.nan)
    s=float(a[q].sum()) if q.any() else 0.0
    if s>0: out[q]=a[q]/s
    return out

def log_binned_pmf(mass,bins_per_decade=8):
    a=finite_positive(mass)
    if len(a)==0:
        return pd.DataFrame(columns=["bin_left","bin_right","bin_center","count","pmf","density"])
    lo=float(a.min()); hi=float(a.max())
    if lo==hi:
        return pd.DataFrame([{
            "bin_left":lo,"bin_right":hi,"bin_center":lo,
            "count":len(a),"pmf":1.0,"density":np.nan
        }])
    decades=np.log10(hi)-np.log10(lo)
    nb=max(4,int(np.ceil(decades*bins_per_decade)))
    edges=np.geomspace(lo,hi*(1+1e-12),nb+1)
    counts,_=np.histogram(a,bins=edges)
    widths=np.diff(edges)
    centers=np.sqrt(edges[:-1]*edges[1:])
    pmf=counts/counts.sum()
    density=counts/(counts.sum()*widths)
    return pd.DataFrame({
        "bin_left":edges[:-1],"bin_right":edges[1:],"bin_center":centers,
        "count":counts.astype(int),"pmf":pmf,"density":density
    })

def empirical_ccdf(mass):
    a=finite_positive(mass)
    if len(a)==0:return pd.DataFrame(columns=["mass","ccdf","rank_desc"])
    x=np.sort(a)
    n=len(x)
    vals, first_idx=np.unique(x,return_index=True)
    ccdf=(n-first_idx)/n
    # descending rank at each unique mass threshold
    rank_desc=n-first_idx
    return pd.DataFrame({"mass":vals,"ccdf":ccdf,"rank_desc":rank_desc.astype(int)})

def rank_size(mass):
    a=finite_positive(mass)
    if len(a)==0:return pd.DataFrame(columns=["rank","mass","mass_fraction"])
    a=np.sort(a)[::-1]
    return pd.DataFrame({
        "rank":np.arange(1,len(a)+1,dtype=int),
        "mass":a,
        "mass_fraction":a/a.sum()
    })

def lorenz_curve(mass):
    a=finite_positive(mass)
    if len(a)==0:return pd.DataFrame(columns=["fraction_supernodes","fraction_mass"])
    a=np.sort(a)
    cum=np.concatenate([[0.0],np.cumsum(a)/a.sum()])
    f=np.linspace(0,1,len(a)+1)
    return pd.DataFrame({"fraction_supernodes":f,"fraction_mass":cum})

def local_loglog_slope(x,y,min_points=4):
    x=np.asarray(x,float); y=np.asarray(y,float)
    q=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
    x=x[q]; y=y[q]
    if len(x)<min_points:return np.nan,np.nan,int(len(x))
    lx=np.log10(x); ly=np.log10(y)
    A=np.column_stack([lx,np.ones(len(lx))])
    beta=np.linalg.lstsq(A,ly,rcond=None)[0]
    pred=A@beta
    ss_res=float(np.sum((ly-pred)**2))
    ss_tot=float(np.sum((ly-ly.mean())**2))
    r2=1-ss_res/ss_tot if ss_tot>0 else np.nan
    return float(beta[0]),float(r2),int(len(x))

def concentration_trajectory(summary):
    cols=[c for c in [
        "landmark_id","minimum_nodes","minimum_removed_fraction",
        "formal_supernodes","effective_domain_number",
        "K50","K80","K90","mass_gini","mass_median","mass_q90",
        "mass_q99","mass_max"
    ] if c in summary.columns]
    return summary[cols].sort_values("minimum_removed_fraction").copy()
