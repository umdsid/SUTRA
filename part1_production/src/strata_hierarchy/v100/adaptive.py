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

def block_for(c):
    for b,p in BLOCKS.items():
        if any(c.startswith(x) for x in p):
            return b
    return None

def robust_scale(x):
    a=np.asarray(x,float); a=a[np.isfinite(a)]
    if len(a)<4:return np.nan,np.nan
    med=float(np.median(a))
    q25,q75=np.quantile(a,[.25,.75])
    s=float((q75-q25)/1.349)
    if not np.isfinite(s) or s<=1e-12:
        s=float(np.median(np.abs(a-med))*1.4826)
    if not np.isfinite(s) or s<=1e-12:
        s=float(np.std(a))
    return med,(s if np.isfinite(s) and s>1e-12 else np.nan)

def calibration_from_v094(v094_norm, v0941_flow, min_nodes=20):
    """Freeze normalization and low-flow thresholds from audited exhaustive runs."""
    norm=v094_norm.copy()
    norm["block"]=[block_for(c) for c in norm.observable]
    norm=norm[norm.block.notna()].copy()

    f=v0941_flow.copy()
    q=(f.nodes>=min_nodes)&np.isfinite(f.block_balanced_velocity)&np.isfinite(f.block_balanced_acceleration)
    if not q.any():
        raise RuntimeError("no eligible audited flow landmarks")
    vt=float(np.quantile(f.loc[q,"block_balanced_velocity"],.35))
    at=float(np.quantile(f.loc[q,"block_balanced_acceleration"],.50))
    hi_v=float(np.quantile(f.loc[q,"block_balanced_velocity"],.80))
    hi_a=float(np.quantile(f.loc[q,"block_balanced_acceleration"],.80))
    return norm,dict(velocity_low=vt,acceleration_low=at,velocity_high=hi_v,acceleration_high=hi_a)

def standardize_landmarks(rows,norm):
    df=pd.DataFrame(rows)
    cols=[]; blocks=[]; Z=[]
    for r in norm.itertuples(index=False):
        c=str(r.observable)
        if c not in df.columns: continue
        x=df[c].to_numpy(float)
        z=(x-float(r.median))/float(r.robust_scale)
        if np.isfinite(z).sum()<3: continue
        z=pd.Series(z).interpolate(limit_direction="both").to_numpy(float)
        cols.append(c);blocks.append(str(r.block));Z.append(z)
    if not Z:
        return [],[],np.empty((len(df),0))
    return cols,blocks,np.stack(Z,axis=1)

def causal_derivative(Y,x,stride):
    Y=np.asarray(Y,float); x=np.asarray(x,float)
    n,p=Y.shape
    out=np.full((n,p),np.nan)
    s=max(1,int(stride))
    for i in range(s,n):
        dx=x[i]-x[i-s]
        if abs(dx)>1e-12:
            out[i]=(Y[i]-Y[i-s])/dx
    return out

def rms(A):
    A=np.asarray(A,float)
    return np.sqrt(np.nanmean(A*A,axis=1))

def block_norms(D,blocks):
    out={}
    for b in sorted(set(blocks)):
        idx=[i for i,x in enumerate(blocks) if x==b]
        if idx:
            out[b]=rms(D[:,idx])
    return out

def equal_block_norm(B):
    if not B:return np.array([])
    M=np.stack(list(B.values()),axis=1)
    return rms(M)

def trailing_median(x,width):
    return pd.Series(np.asarray(x,float)).rolling(int(width),min_periods=1).median().to_numpy(float)

def online_metric(rows,norm,strides=(1,2,3),windows=(3,5,7)):
    if len(rows)<8:
        return None
    df=pd.DataFrame(rows)
    ell=df.ell.to_numpy(float)
    cols,blocks,Z=standardize_landmarks(rows,norm)
    if Z.shape[1]<6:return None

    configs=[]
    block_last={}
    for st in strides:
        V=causal_derivative(Z,ell,st)
        A=causal_derivative(V,ell,st)
        BV=block_norms(V,blocks); BA=block_norms(A,blocks)
        v=equal_block_norm(BV); a=equal_block_norm(BA)
        for w in windows:
            vs=trailing_median(v,w); ac=trailing_median(a,w)
            configs.append((st,w,float(vs[-1]),float(ac[-1])))
        for b in BV:
            block_last[b]=float(BV[b][-1]) if np.isfinite(BV[b][-1]) else np.nan

    return {
        "configs":configs,
        "block_velocity":block_last,
        "observables":len(cols),
        "blocks":len(set(blocks))
    }

def classify_online(metric,thresholds):
    if metric is None:
        return {
            "stable_support":0.0,"high_support":0.0,
            "median_velocity":np.nan,"median_acceleration":np.nan,
            "state":"WARMUP"
        }
    cfg=metric["configs"]
    vals=np.array([[x[2],x[3]] for x in cfg],float)
    low=(vals[:,0]<=thresholds["velocity_low"])&(vals[:,1]<=thresholds["acceleration_low"])
    high=(vals[:,0]>=thresholds["velocity_high"])|(vals[:,1]>=thresholds["acceleration_high"])
    ss=float(low.mean()); hs=float(high.mean())
    if hs>=.50:state="FAST_CHANGE"
    elif ss>=.60:state="LOW_FLOW"
    else:state="REGULAR"
    return {
        "stable_support":ss,"high_support":hs,
        "median_velocity":float(np.nanmedian(vals[:,0])),
        "median_acceleration":float(np.nanmedian(vals[:,1])),
        "state":state
    }

def choose_fraction(classification,cfg,confirming=False):
    if confirming:
        return float(cfg["fraction_confirm"]), "confirmation_ultraslow"
    state=classification["state"]
    if state=="FAST_CHANGE":
        return float(cfg["fraction_min"]), "adaptive_ultraslow"
    if state=="LOW_FLOW":
        return float(cfg["fraction_min"]), "low_flow_probe"
    if state=="WARMUP":
        return float(cfg["fraction_base"]), "warmup_base"
    return float(cfg["fraction_base"]), "adaptive_base"

class RegimeDetector:
    """Causal persistent-regime detector with independent slow confirmation."""
    def __init__(self,cfg):
        self.cfg=cfg
        self.low_run=0
        self.confirm_run=0
        self.confirming=False
        self.entry_index=None
        self.best_index=None
        self.best_velocity=np.inf

    def update(self,index,classification,nodes,removed):
        eligible=(nodes>=int(self.cfg["min_effective_nodes"])
                  and removed>=float(self.cfg["min_removed_fraction"])
                  and removed<=float(self.cfg["max_removed_fraction"]))
        stable=eligible and classification["stable_support"]>=float(self.cfg["stable_support_required"])
        if not self.confirming:
            self.low_run=self.low_run+1 if stable else 0
            if stable and classification["median_velocity"]<self.best_velocity:
                self.best_velocity=classification["median_velocity"];self.best_index=index
            if self.low_run>=int(self.cfg["persistence_landmarks"]):
                self.confirming=True
                self.entry_index=index-self.low_run+1
                self.confirm_run=0
                return "ENTER_CONFIRMATION"
            return "CONTINUE"
        else:
            if stable:
                self.confirm_run+=1
                if classification["median_velocity"]<self.best_velocity:
                    self.best_velocity=classification["median_velocity"];self.best_index=index
            else:
                # failed confirmation: resume search without stopping.
                self.confirming=False;self.low_run=0;self.confirm_run=0
                self.entry_index=None;self.best_index=None;self.best_velocity=np.inf
                return "CONFIRMATION_FAILED"
            if self.confirm_run>=int(self.cfg["confirmation_landmarks"]):
                return "STOP_CONFIRMED_REGIME"
            return "CONFIRMING"
