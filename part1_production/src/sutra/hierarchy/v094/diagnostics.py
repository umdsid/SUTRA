from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

IDENTITY_COLUMNS={"sample","landmark_index","microstep","removed_fraction","nodes","superedges"}
PREFERRED_PREFIXES=(
    "expression_","functional_","cellchat_","tension_","pressure_",
    "component","cycle_rank","mean_degree","max_degree","directional_",
    "transport_","geodesic_","holonomy_","supernode_"
)

def load_landmark_table(landmark_dir):
    paths=sorted(Path(landmark_dir).glob("landmark_*.json"))
    if not paths:
        raise FileNotFoundError(f"no landmark json files in {landmark_dir}")
    rows=[json.loads(p.read_text()) for p in paths]
    return pd.DataFrame(rows).sort_values("landmark_index").reset_index(drop=True)

def natural_scale(n0,nodes):
    n=np.asarray(nodes,float)
    return np.log(float(n0)/np.maximum(n,1.0))

def robust_center_scale(x):
    a=np.asarray(x,float)
    q=np.isfinite(a)
    if q.sum()<4:
        return np.nan,np.nan
    z=a[q]
    med=float(np.median(z))
    q25,q75=np.quantile(z,[.25,.75])
    s=float((q75-q25)/1.349)
    if not np.isfinite(s) or s<=1e-12:
        s=float(np.median(np.abs(z-med))*1.4826)
    if not np.isfinite(s) or s<=1e-12:
        s=float(np.std(z))
    if not np.isfinite(s) or s<=1e-12:
        return med,np.nan
    return med,s

def choose_observables(df,min_finite_fraction=.65):
    cols=[]
    for c in df.columns:
        if c in IDENTITY_COLUMNS or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if not c.startswith(PREFERRED_PREFIXES):
            continue
        x=df[c].to_numpy(float)
        if np.isfinite(x).mean()<min_finite_fraction:
            continue
        _,s=robust_center_scale(x)
        if np.isfinite(s):
            cols.append(c)
    return cols

def robust_standardize(df,cols):
    Z=np.full((len(df),len(cols)),np.nan,float)
    meta=[]
    for j,c in enumerate(cols):
        x=df[c].to_numpy(float)
        xs=pd.Series(x).interpolate(limit_area="inside").to_numpy(float)
        med,s=robust_center_scale(xs)
        Z[:,j]=(xs-med)/s
        meta.append({"observable":c,"median":med,"robust_scale":s})
    return Z,pd.DataFrame(meta)

def finite_gradient(Y,x):
    Y=np.asarray(Y,float); x=np.asarray(x,float)
    out=np.full_like(Y,np.nan)
    for j in range(Y.shape[1]):
        y=Y[:,j]
        q=np.isfinite(y)&np.isfinite(x)
        if q.sum()<3: continue
        yy=pd.Series(y).interpolate(limit_direction="both").to_numpy(float)
        out[:,j]=np.gradient(yy,x,edge_order=1)
    return out

def row_rms(A):
    A=np.asarray(A,float)
    return np.sqrt(np.nanmean(A*A,axis=1))

def smooth_median(x,width=5):
    return pd.Series(np.asarray(x,float)).rolling(width,center=True,min_periods=1).median().to_numpy(float)

def contiguous_true_runs(mask):
    m=np.asarray(mask,bool)
    runs=[]; start=None
    for i,v in enumerate(m):
        if v and start is None:
            start=i
        if start is not None and ((not v) or i==len(m)-1):
            end=i if (v and i==len(m)-1) else i-1
            runs.append((start,end)); start=None
    return runs

def local_change_points(velocity,acceleration,min_separation=3,top_k=12):
    v=np.asarray(velocity,float); a=np.asarray(acceleration,float)
    vm,vs=robust_center_scale(v); am,as_=robust_center_scale(a)
    vz=np.maximum(0,(v-vm)/(vs if np.isfinite(vs) else 1.))
    az=np.maximum(0,(a-am)/(as_ if np.isfinite(as_) else 1.))
    score=vz+az
    candidates=[]
    for i in range(1,len(score)-1):
        if np.isfinite(score[i]) and score[i]>=score[i-1] and score[i]>=score[i+1]:
            candidates.append((float(score[i]),i))
    candidates.sort(reverse=True)
    chosen=[]
    for s,i in candidates:
        if all(abs(i-j)>=min_separation for _,j in chosen):
            chosen.append((s,i))
        if len(chosen)>=top_k:
            break
    return sorted(chosen,key=lambda z:z[1])

def identify_stationary_windows(
    df,velocity,acceleration,min_nodes=20,min_removed=.05,max_removed=.995,
    velocity_quantile=.35,acceleration_quantile=.50,min_landmarks=5
):
    eligible=(
        (df.nodes.to_numpy(float)>=min_nodes)
        &(df.removed_fraction.to_numpy(float)>=min_removed)
        &(df.removed_fraction.to_numpy(float)<=max_removed)
    )
    v=np.asarray(velocity,float); a=np.asarray(acceleration,float)
    q=eligible&np.isfinite(v)&np.isfinite(a)
    if q.sum()<min_landmarks:
        return [],np.zeros(len(df),bool),np.nan,np.nan
    vthr=float(np.quantile(v[q],velocity_quantile))
    athr=float(np.quantile(a[q],acceleration_quantile))
    stable=q&(v<=vthr)&(a<=athr)
    runs=[(s,e) for s,e in contiguous_true_runs(stable) if e-s+1>=min_landmarks]
    windows=[]
    ell=df.ell.to_numpy(float)
    for s,e in runs:
        length=e-s+1
        span=float(ell[e]-ell[s]) if e>s else 0.
        meanv=float(np.mean(v[s:e+1])); meana=float(np.mean(a[s:e+1]))
        score=float((span+1e-9)*length/((meanv+1e-9)*(meana+1e-9)))
        windows.append({
            "start_index":int(s),"end_index":int(e),"landmarks":int(length),
            "ell_start":float(ell[s]),"ell_end":float(ell[e]),"ell_span":span,
            "removed_start":float(df.removed_fraction.iloc[s]),
            "removed_end":float(df.removed_fraction.iloc[e]),
            "nodes_start":int(df.nodes.iloc[s]),"nodes_end":int(df.nodes.iloc[e]),
            "velocity_mean":meanv,"acceleration_mean":meana,
            "persistence_score":score
        })
    windows.sort(key=lambda d:d["persistence_score"],reverse=True)
    return windows,stable,vthr,athr

def recommendations(df,windows,velocity,acceleration):
    rec=[]
    for rank,w in enumerate(windows[:5],1):
        s,e=w["start_index"],w["end_index"]
        q=np.arange(s,e+1)
        j=int(q[np.nanargmin(np.asarray(velocity)[q])])
        rec.append({
            "rank":rank,
            "window_start_landmark":int(df.landmark_index.iloc[s]),
            "window_end_landmark":int(df.landmark_index.iloc[e]),
            "representative_landmark":int(df.landmark_index.iloc[j]),
            "nodes":int(df.nodes.iloc[j]),
            "removed_fraction":float(df.removed_fraction.iloc[j]),
            "ell":float(df.ell.iloc[j]),
            "velocity":float(velocity[j]),
            "acceleration":float(acceleration[j]),
            "reason":"persistent low effective-state flow with nontrivial spatial tessellation"
        })
    return rec

def analyze_landmarks(df,n0,cfg):
    d=df.copy()
    d["ell"]=natural_scale(n0,d.nodes)
    cols=choose_observables(d,float(cfg["min_finite_fraction"]))
    Z,meta=robust_standardize(d,cols)
    V=finite_gradient(Z,d.ell.to_numpy(float))
    A=finite_gradient(V,d.ell.to_numpy(float))
    speed=smooth_median(row_rms(V),int(cfg["smoothing_width"]))
    accel=smooth_median(row_rms(A),int(cfg["smoothing_width"]))

    windows,stable,vthr,athr=identify_stationary_windows(
        d,speed,accel,
        min_nodes=int(cfg["min_effective_nodes"]),
        min_removed=float(cfg["min_removed_fraction"]),
        max_removed=float(cfg["max_removed_fraction_for_biological_scale"]),
        velocity_quantile=float(cfg["velocity_quantile"]),
        acceleration_quantile=float(cfg["acceleration_quantile"]),
        min_landmarks=int(cfg["min_stationary_landmarks"])
    )
    cps=local_change_points(
        speed,accel,
        min_separation=int(cfg["change_point_min_separation"]),
        top_k=int(cfg["change_point_top_k"])
    )
    cp_rows=[]
    for score,i in cps:
        cp_rows.append({
            "landmark_index":int(d.landmark_index.iloc[i]),
            "table_index":int(i),
            "nodes":int(d.nodes.iloc[i]),
            "removed_fraction":float(d.removed_fraction.iloc[i]),
            "ell":float(d.ell.iloc[i]),
            "velocity":float(speed[i]),
            "acceleration":float(accel[i]),
            "change_score":float(score)
        })
    d["flow_velocity_norm"]=speed
    d["flow_acceleration_norm"]=accel
    d["stationary_candidate"]=stable
    rec=recommendations(d,windows,speed,accel)
    info={
        "n_observables":len(cols),
        "velocity_threshold":vthr,
        "acceleration_threshold":athr,
        "n_stationary_windows":len(windows),
        "observable_columns":cols
    }
    return d,meta,pd.DataFrame(windows),pd.DataFrame(cp_rows),pd.DataFrame(rec),info
