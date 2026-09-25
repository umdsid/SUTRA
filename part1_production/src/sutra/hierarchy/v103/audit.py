from __future__ import annotations
import math
import numpy as np
import pandas as pd

BLOCKS={
    "expression":("expression_",),
    "functional_GO_MSIGDB":("functional_",),
    "CellChat":("cellchat_",),
    "mechanics":("tension_",),
    "pressure":("pressure_",),
    "topology":("component","cycle_rank","mean_degree","max_degree","supernode_"),
    "directional_geometry":("directional_",),
    "transport":("transport_",),
    "geodesics":("geodesic_",),
    "holonomy":("holonomy_",),
}

def block_for(c):
    for b,p in BLOCKS.items():
        if any(str(c).startswith(x) for x in p):
            return b
    return None

def frozen_coordinates(df,norm,min_finite=.65):
    cols=[]; blocks=[]; zz=[]
    for r in norm.itertuples(index=False):
        c=str(r.observable); b=block_for(c)
        if b is None or c not in df.columns: continue
        x=df[c].to_numpy(float)
        if np.isfinite(x).mean()<min_finite: continue
        s=float(r.robust_scale)
        if not np.isfinite(s) or s<=1e-12: continue
        cols.append(c); blocks.append(b); zz.append((x-float(r.median))/s)
    return cols,blocks,(np.stack(zz,axis=1) if zz else np.empty((len(df),0)))

def block_displacements(Z,blocks,lags):
    expected=sorted(set(blocks))
    out=[]
    for lag in lags:
        lag=int(lag)
        for i in range(lag,len(Z)):
            for b in expected:
                idx=[j for j,x in enumerate(blocks) if x==b]
                a=Z[i,idx]; c=Z[i-lag,idx]
                q=np.isfinite(a)&np.isfinite(c)
                d=np.nan
                if q.any():
                    z=a[q]-c[q]
                    d=float(np.sqrt(np.mean(z*z)))
                out.append({
                    "table_index":int(i),"lag":lag,"block":b,
                    "displacement":d,"resolved":bool(np.isfinite(d))
                })
    return pd.DataFrame(out)

def hist_median(x,end,width):
    lo=max(0,int(end)-int(width))
    z=np.asarray(x[lo:int(end)],float); z=z[np.isfinite(z)]
    return float(np.median(z)) if len(z) else np.nan

def ratio(a,b):
    return float(a/b) if np.isfinite(a) and np.isfinite(b) and b>1e-12 else np.nan

def block_closure_ratios(block_disp,df,recent_width,history_width):
    rows=[]
    for (block,lag),g in block_disp.groupby(["block","lag"]):
        arr=np.full(len(df),np.nan)
        for r in g.itertuples(index=False):
            arr[int(r.table_index)]=float(r.displacement)
        for i in range(len(df)):
            hist_end=max(0,i-int(recent_width)+1)
            cur=arr[i]
            prior=hist_median(arr,hist_end,history_width)
            rows.append({
                "table_index":i,"block":block,"lag":int(lag),
                "current_displacement":float(cur) if np.isfinite(cur) else np.nan,
                "historical_displacement":prior,
                "closure_ratio":ratio(cur,prior)
            })
    return pd.DataFrame(rows)

def gate_table(features,cfg):
    f=features.copy()
    f["pass_velocity"]=np.isfinite(f.velocity_ratio)&(f.velocity_ratio<=float(cfg["velocity_ratio_max"]))
    f["pass_acceleration"]=np.isfinite(f.acceleration_ratio)&(f.acceleration_ratio<=float(cfg["acceleration_ratio_max"]))
    f["pass_closure"]=f.closure_support>=float(cfg["closure_support_required"])
    f["pass_resolution"]=f.resolved_closure_lags>=int(cfg["min_resolved_closure_lags"])
    f["pass_eligible"]=f.eligible.astype(bool)
    f["gate_count"]=(
        f.pass_velocity.astype(int)+f.pass_acceleration.astype(int)+
        f.pass_closure.astype(int)+f.pass_resolution.astype(int)
    )
    f["near_plateau_3of4"]=f.gate_count>=3
    f["near_plateau_2of4"]=f.gate_count>=2

    # Dimensionless "distance to satisfying gates"; lower is more plateau-like.
    vr=np.where(np.isfinite(f.velocity_ratio),f.velocity_ratio/float(cfg["velocity_ratio_max"]),np.nan)
    ar=np.where(np.isfinite(f.acceleration_ratio),f.acceleration_ratio/float(cfg["acceleration_ratio_max"]),np.nan)
    cr=np.where(np.isfinite(f.closure_median_ratio),f.closure_median_ratio/float(cfg["closure_ratio_max"]),np.nan)
    M=np.vstack([vr,ar,cr]).T
    finite=np.isfinite(M)
    cnt=finite.sum(axis=1)
    sq=np.where(finite,M*M,0.).sum(axis=1)
    score=np.full(len(f),np.nan)
    q=cnt>0
    score[q]=np.sqrt(sq[q]/cnt[q])
    f["plateau_distance_score"]=score
    return f

def local_minima(x,min_sep=4,top_k=12):
    x=np.asarray(x,float)
    cand=[]
    for i in range(1,len(x)-1):
        if np.isfinite(x[i]) and x[i]<=x[i-1] and x[i]<=x[i+1]:
            cand.append((float(x[i]),i))
    cand.sort()
    chosen=[]
    for s,i in cand:
        if all(abs(i-j)>=min_sep for _,j in chosen):
            chosen.append((s,i))
        if len(chosen)>=top_k: break
    return sorted(chosen,key=lambda z:z[1])

def metastable_candidates(gates,df,cfg):
    mins=local_minima(
        gates.plateau_distance_score.to_numpy(float),
        int(cfg["metastable_min_separation"]),
        int(cfg["metastable_top_k"])
    )
    rows=[]
    for score,i in mins:
        r=gates.iloc[i]
        rows.append({
            "landmark_index":int(r.landmark_index),"table_index":int(i),
            "nodes":int(r.nodes),"removed_fraction":float(r.removed_fraction),
            "plateau_distance_score":float(score),
            "gate_count":int(r.gate_count),
            "pass_velocity":bool(r.pass_velocity),
            "pass_acceleration":bool(r.pass_acceleration),
            "pass_closure":bool(r.pass_closure),
            "pass_resolution":bool(r.pass_resolution),
            "active":bool(r.active),
            "plateau":bool(r.plateau),
        })
    return pd.DataFrame(rows)

def veto_summary(gates):
    eligible=gates[gates.pass_eligible.astype(bool)]
    rows=[]
    for name in ["pass_velocity","pass_acceleration","pass_closure","pass_resolution"]:
        rows.append({
            "gate":name.replace("pass_",""),
            "eligible_landmarks":int(len(eligible)),
            "pass_count":int(eligible[name].sum()) if len(eligible) else 0,
            "pass_fraction":float(eligible[name].mean()) if len(eligible) else np.nan,
            "veto_count":int((~eligible[name].astype(bool)).sum()) if len(eligible) else 0,
        })
    return pd.DataFrame(rows)

def block_state_summary(block_ratios,gates,cfg):
    rows=[]
    near=set(gates.loc[gates.near_plateau_2of4,"table_index"].astype(int))
    for (block,lag),g in block_ratios.groupby(["block","lag"]):
        z=g[g.table_index.isin(near)]
        vals=z.closure_ratio.to_numpy(float)
        vals=vals[np.isfinite(vals)]
        rows.append({
            "block":block,"lag":int(lag),"near_plateau_landmarks":int(len(z)),
            "resolved":int(len(vals)),
            "median_closure_ratio":float(np.median(vals)) if len(vals) else np.nan,
            "q75_closure_ratio":float(np.quantile(vals,.75)) if len(vals) else np.nan,
            "stable_fraction":float(np.mean(vals<=float(cfg["closure_ratio_max"]))) if len(vals) else np.nan,
        })
    return pd.DataFrame(rows)

def unstable_block_at_candidates(block_ratios,candidates,cfg):
    rows=[]
    for c in candidates.itertuples(index=False):
        i=int(c.table_index)
        z=block_ratios[block_ratios.table_index==i]
        for block,g in z.groupby("block"):
            vals=g.closure_ratio.to_numpy(float)
            vals=vals[np.isfinite(vals)]
            med=float(np.median(vals)) if len(vals) else np.nan
            rows.append({
                "landmark_index":int(c.landmark_index),"nodes":int(c.nodes),
                "removed_fraction":float(c.removed_fraction),"block":block,
                "median_multilag_closure_ratio":med,
                "stable_multilag":bool(np.isfinite(med) and med<=float(cfg["closure_ratio_max"]))
            })
    out=pd.DataFrame(rows)
    if len(out):
        out["block_rank_unstable"]=out.groupby("landmark_index")["median_multilag_closure_ratio"].rank(
            method="dense",ascending=False
        )
    return out
