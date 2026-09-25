from __future__ import annotations
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

IDENTITY={"sample","landmark_index","microstep","removed_fraction","nodes","superedges",
          "ell","flow_velocity_norm","flow_acceleration_norm","stationary_candidate"}

def block_for_column(c):
    for b,prefixes in BLOCKS.items():
        if any(c.startswith(p) for p in prefixes):
            return b
    return None

def robust_center_scale(x):
    a=np.asarray(x,float)
    q=np.isfinite(a)
    if q.sum()<4:return np.nan,np.nan
    z=a[q]; med=float(np.median(z))
    q25,q75=np.quantile(z,[.25,.75]); s=float((q75-q25)/1.349)
    if not np.isfinite(s) or s<=1e-12:
        s=float(np.median(np.abs(z-med))*1.4826)
    if not np.isfinite(s) or s<=1e-12:
        s=float(np.std(z))
    return (med,s) if np.isfinite(s) and s>1e-12 else (med,np.nan)

def choose_columns(df,min_finite=.65):
    out={}
    for c in df.columns:
        if c in IDENTITY or not pd.api.types.is_numeric_dtype(df[c]): continue
        b=block_for_column(c)
        if b is None: continue
        x=df[c].to_numpy(float)
        if np.isfinite(x).mean()<min_finite: continue
        _,s=robust_center_scale(x)
        if np.isfinite(s): out[c]=b
    return out

def standardize(df,colmap):
    cols=list(colmap)
    Z=np.full((len(df),len(cols)),np.nan)
    for j,c in enumerate(cols):
        x=pd.Series(df[c].to_numpy(float)).interpolate(limit_direction="both").to_numpy(float)
        med,s=robust_center_scale(x)
        Z[:,j]=(x-med)/s
    return cols,Z

def stride_gradient(Y,x,stride=1):
    Y=np.asarray(Y,float); x=np.asarray(x,float)
    n,p=Y.shape; out=np.full((n,p),np.nan)
    stride=max(1,int(stride))
    for i in range(n):
        lo=max(0,i-stride); hi=min(n-1,i+stride)
        if hi==lo: continue
        dx=x[hi]-x[lo]
        if abs(dx)<=1e-12: continue
        out[i]=(Y[hi]-Y[lo])/dx
    return out

def smooth(x,width):
    return pd.Series(np.asarray(x,float)).rolling(
        int(width),center=True,min_periods=1
    ).median().to_numpy(float)

def block_flow(V,cols,colmap):
    rows={}
    for b in BLOCKS:
        idx=[j for j,c in enumerate(cols) if colmap[c]==b]
        if not idx: continue
        A=V[:,idx]
        rows[b]=np.sqrt(np.nanmean(A*A,axis=1))
    return rows

def balanced_norm(block_series):
    if not block_series:
        return np.array([])
    M=np.stack(list(block_series.values()),axis=1)
    return np.sqrt(np.nanmean(M*M,axis=1))

def unbalanced_norm(V):
    return np.sqrt(np.nanmean(V*V,axis=1))

def contiguous_runs(m):
    m=np.asarray(m,bool); runs=[]; s=None
    for i,v in enumerate(m):
        if v and s is None:s=i
        if s is not None and ((not v) or i==len(m)-1):
            e=i if v and i==len(m)-1 else i-1
            runs.append((s,e));s=None
    return runs

def stationary_windows(df,v,a,cfg):
    eligible=(
        (df.nodes.to_numpy(float)>=cfg["min_effective_nodes"])
        &(df.removed_fraction.to_numpy(float)>=cfg["min_removed_fraction"])
        &(df.removed_fraction.to_numpy(float)<=cfg["max_removed_fraction"])
    )
    q=eligible&np.isfinite(v)&np.isfinite(a)
    if q.sum()<cfg["min_stationary_landmarks"]:
        return []
    vt=float(np.quantile(v[q],cfg["velocity_quantile"]))
    at=float(np.quantile(a[q],cfg["acceleration_quantile"]))
    mask=q&(v<=vt)&(a<=at)
    out=[]
    for s,e in contiguous_runs(mask):
        if e-s+1<cfg["min_stationary_landmarks"]:continue
        j=s+int(np.nanargmin(v[s:e+1]))
        out.append({
            "start":s,"end":e,"representative":j,
            "ell_start":float(df.ell.iloc[s]),"ell_end":float(df.ell.iloc[e]),
            "ell_rep":float(df.ell.iloc[j]),
            "nodes_rep":int(df.nodes.iloc[j]),
            "removed_rep":float(df.removed_fraction.iloc[j]),
            "velocity_rep":float(v[j]),"acceleration_rep":float(a[j]),
            "span":float(df.ell.iloc[e]-df.ell.iloc[s]),
        })
    return out

def window_overlap(a,b):
    lo=max(a["ell_start"],b["ell_start"]); hi=min(a["ell_end"],b["ell_end"])
    inter=max(0.,hi-lo)
    union=max(a["ell_end"],b["ell_end"])-min(a["ell_start"],b["ell_start"])
    return inter/union if union>0 else 0.

def compute_metric(df,cfg,smoothing,stride,balanced=True):
    colmap=choose_columns(df,cfg["min_finite_fraction"])
    cols,Z=standardize(df,colmap)
    ell=df.ell.to_numpy(float)
    V=stride_gradient(Z,ell,stride)
    A=stride_gradient(V,ell,stride)
    bv=block_flow(V,cols,colmap); ba=block_flow(A,cols,colmap)
    if balanced:
        vel=balanced_norm(bv); acc=balanced_norm(ba)
    else:
        vel=unbalanced_norm(V); acc=unbalanced_norm(A)
    vel=smooth(vel,smoothing); acc=smooth(acc,smoothing)
    wins=stationary_windows(df,vel,acc,cfg)
    return vel,acc,wins,bv,ba,colmap

def contribution_table(df,vel,bv,landmarks=None):
    rows=[]
    n=len(df)
    idx=range(n) if landmarks is None else landmarks
    for i in idx:
        denom=sum(float(bv[b][i])**2 for b in bv if np.isfinite(bv[b][i]))
        for b in bv:
            z=float(bv[b][i])
            frac=(z*z/denom) if denom>0 and np.isfinite(z) else np.nan
            rows.append({
                "landmark_index":int(df.landmark_index.iloc[i]),
                "table_index":int(i),
                "nodes":int(df.nodes.iloc[i]),
                "removed_fraction":float(df.removed_fraction.iloc[i]),
                "ell":float(df.ell.iloc[i]),
                "block":b,
                "block_velocity_rms":z,
                "squared_velocity_fraction":frac,
            })
    return pd.DataFrame(rows)

def audit_grid(df,cfg):
    records=[]; allwins={}
    for balanced in [False,True]:
        mode="block_balanced" if balanced else "observable_balanced"
        for sm in cfg["smoothing_widths"]:
            for st in cfg["derivative_strides"]:
                vel,acc,wins,bv,ba,colmap=compute_metric(
                    df,cfg,int(sm),int(st),balanced=balanced
                )
                key=(mode,int(sm),int(st))
                allwins[key]=wins
                for rank,w in enumerate(sorted(wins,key=lambda z:(-z["span"],z["velocity_rep"])),1):
                    records.append({
                        "mode":mode,"smoothing_width":int(sm),"derivative_stride":int(st),
                        "rank":rank,**w
                    })
    return pd.DataFrame(records),allwins

def baseline_stability(grid,allwins,cfg):
    key=("block_balanced",int(cfg["baseline_smoothing_width"]),int(cfg["baseline_derivative_stride"]))
    base=allwins.get(key,[])
    rows=[]
    configs=list(allwins.keys())
    for rank,b in enumerate(base,1):
        matches=0; ovs=[]; dell=[]
        for k in configs:
            ws=allwins[k]
            if not ws: continue
            best=max(ws,key=lambda w:window_overlap(b,w))
            ov=window_overlap(b,best)
            if ov>=cfg["min_window_overlap_for_match"]:
                matches+=1
            ovs.append(ov); dell.append(abs(best["ell_rep"]-b["ell_rep"]))
        rows.append({
            "baseline_rank":rank,
            "nodes":b["nodes_rep"],
            "removed_fraction":b["removed_rep"],
            "ell":b["ell_rep"],
            "support_fraction":matches/max(len(configs),1),
            "mean_best_window_overlap":float(np.mean(ovs)) if ovs else 0.,
            "median_rep_ell_drift":float(np.median(dell)) if dell else np.nan,
            "stable":bool(matches/max(len(configs),1)>=cfg["min_config_support_fraction"]),
        })
    return pd.DataFrame(rows)

def change_point_indices(df,top_k=12):
    v=df.flow_velocity_norm.to_numpy(float)
    a=df.flow_acceleration_norm.to_numpy(float)
    score=np.nan_to_num(v/np.nanmedian(v))+np.nan_to_num(a/np.nanmedian(a))
    idx=np.argsort(score)[::-1]
    keep=[]
    for i in idx:
        if i<=0 or i>=len(df)-1:continue
        if all(abs(int(i)-j)>=3 for j in keep):
            keep.append(int(i))
        if len(keep)>=top_k:break
    return sorted(keep)
