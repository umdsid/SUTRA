from __future__ import annotations
import math, inspect
import numpy as np
import pandas as pd

BLOCKS={
 "expression":("expression_",),"functional_GO_MSIGDB":("functional_",),
 "CellChat":("cellchat_",),"mechanics":("tension_",),"pressure":("pressure_",),
 "topology":("component","cycle_rank","mean_degree","max_degree","supernode_"),
 "directional_geometry":("directional_",),"transport":("transport_",),
 "geodesics":("geodesic_",),"holonomy":("holonomy_",),
}

def block_for(c):
    for b,p in BLOCKS.items():
        if any(str(c).startswith(x) for x in p): return b
    return None

def finite_rms(a):
    a=np.asarray(a,float); q=np.isfinite(a)
    if not q.any(): return np.nan
    return float(np.sqrt(np.mean(a[q]*a[q])))

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

def block_disp(z1,z0,blocks,min_block_fraction=.7,min_blocks=6):
    expected=sorted(set(blocks)); vals=[]; nr=0
    for b in expected:
        idx=[j for j,x in enumerate(blocks) if x==b]
        q=np.isfinite(z1[idx])&np.isfinite(z0[idx])
        if not q.any(): vals.append(np.nan); continue
        d=z1[idx][q]-z0[idx][q]
        vals.append(float(np.sqrt(np.mean(d*d)))); nr+=1
    need=max(int(min_blocks),int(math.ceil(min_block_fraction*len(expected))))
    return (finite_rms(vals) if nr>=need else np.nan),nr,len(expected)

def causal_displacements(Z,blocks,lags,min_block_fraction,min_blocks):
    out={int(l):np.full(len(Z),np.nan) for l in lags}
    for lag in lags:
        lag=int(lag)
        for i in range(lag,len(Z)):
            out[lag][i]=block_disp(Z[i],Z[i-lag],blocks,min_block_fraction,min_blocks)[0]
    return out

def hist_median(x,end,width):
    lo=max(0,int(end)-int(width))
    z=np.asarray(x[lo:int(end)],float); z=z[np.isfinite(z)]
    return float(np.median(z)) if len(z) else np.nan

def recent_median(x,i,width):
    lo=max(0,int(i)-int(width)+1)
    z=np.asarray(x[lo:int(i)+1],float); z=z[np.isfinite(z)]
    return float(np.median(z)) if len(z) else np.nan

def ratio(a,b):
    return float(a/b) if np.isfinite(a) and np.isfinite(b) and b>1e-12 else np.nan

def one_step_velocity(d1,ell):
    out=np.full(len(ell),np.nan)
    for i in range(1,len(ell)):
        de=ell[i]-ell[i-1]
        if np.isfinite(d1[i]) and de>1e-12: out[i]=d1[i]/de
    return out

def acceleration(v,ell):
    out=np.full(len(ell),np.nan)
    for i in range(2,len(ell)):
        de=ell[i]-ell[i-1]
        if np.isfinite(v[i]) and np.isfinite(v[i-1]) and de>1e-12:
            out[i]=abs(v[i]-v[i-1])/de
    return out

def build_future_blind_features(df,norm,cfg,recent_width,history_width):
    """Every feature at i uses only landmarks <= i."""
    cols,blocks,Z=frozen_coordinates(df,norm,float(cfg["min_finite_fraction"]))
    lags=[int(x) for x in cfg["closure_lags"]]
    D=causal_displacements(
        Z,blocks,lags,float(cfg["min_resolved_block_fraction"]),
        int(cfg["min_resolved_blocks_absolute"])
    )
    ell=df.ell.to_numpy(float)
    vel=one_step_velocity(D[1],ell); acc=acceleration(vel,ell)
    rows=[]
    for i in range(len(df)):
        hist_end=max(0,i-int(recent_width)+1)
        rv=ratio(recent_median(vel,i,recent_width),hist_median(vel,hist_end,history_width))
        ra=ratio(recent_median(acc,i,recent_width),hist_median(acc,hist_end,history_width))
        cr=[]
        for lag in lags:
            cur=D[lag][i]
            cr.append(ratio(cur,hist_median(D[lag],hist_end,history_width)))
        vals=np.array([x for x in cr if np.isfinite(x)],float)
        ncr=len(vals)
        cmed=float(np.median(vals)) if ncr else np.nan
        csup=float(np.mean(vals<=float(cfg["closure_ratio_max"]))) if ncr else 0.
        slowdown=(np.isfinite(rv) and np.isfinite(ra)
                  and rv<=float(cfg["velocity_ratio_max"])
                  and ra<=float(cfg["acceleration_ratio_max"]))
        active=((np.isfinite(rv) and rv>=float(cfg["active_velocity_ratio_min"]))
                or (np.isfinite(cmed) and cmed>=float(cfg["active_closure_ratio_min"])))
        eligible=(int(df.nodes.iloc[i])>=int(cfg["min_effective_nodes"])
                  and float(df.removed_fraction.iloc[i])>=float(cfg["detector_min_removed_fraction"])
                  and float(df.removed_fraction.iloc[i])<=float(cfg["detector_max_removed_fraction"])
                  and ncr>=int(cfg["min_resolved_closure_lags"])
                  and np.isfinite(rv) and np.isfinite(ra))
        plateau=eligible and slowdown and csup>=float(cfg["closure_support_required"])
        rows.append({
          "landmark_index":int(df.landmark_index.iloc[i]),"table_index":i,
          "nodes":int(df.nodes.iloc[i]),"removed_fraction":float(df.removed_fraction.iloc[i]),
          "ell":float(df.ell.iloc[i]),"recent_width":int(recent_width),"history_width":int(history_width),
          "velocity_ratio":rv,"acceleration_ratio":ra,"closure_median_ratio":cmed,
          "closure_support":csup,"resolved_closure_lags":int(ncr),
          "active":bool(active),"eligible":bool(eligible),"plateau":bool(plateau)
        })
    return pd.DataFrame(rows),{"observables":len(cols),"blocks":sorted(set(blocks)),"n_blocks":len(set(blocks))}

class PlateauDetector:
    def __init__(self,cfg):
        self.cfg=cfg; self.run=0; self.confirm=0; self.confirming=False
        self.entry=None; self.best=None; self.best_score=np.inf
    def prior_active(self,f,i):
        lo=max(0,i-int(self.cfg["entry_history_landmarks"]))
        hi=max(lo,i-self.run+1)
        n=int(f.iloc[lo:hi].active.sum()) if hi>lo else 0
        return n>=int(self.cfg["min_active_landmarks_before_entry"])
    def update(self,f,i):
        r=f.iloc[i]; p=bool(r.plateau)
        if not self.confirming:
            self.run=self.run+1 if p else 0
            if p:
                score=np.nanmedian([r.velocity_ratio,r.acceleration_ratio,r.closure_median_ratio])
                if np.isfinite(score) and score<self.best_score:
                    self.best_score=float(score); self.best=int(r.landmark_index)
            if self.run>=int(self.cfg["persistence_landmarks"]):
                if not self.prior_active(f,i): return "PLATEAU_WITHOUT_ENTRY"
                self.confirming=True; self.confirm=0
                self.entry=int(r.landmark_index)-self.run+1
                return "ENTER_CONFIRMATION"
            return "CONTINUE"
        if p:
            self.confirm+=1
            if self.confirm>=int(self.cfg["confirmation_landmarks"]):
                return "STOP_CONFIRMED_PLATEAU"
            return "CONFIRMING"
        self.run=0; self.confirm=0; self.confirming=False
        self.entry=None; self.best=None; self.best_score=np.inf
        return "CONFIRMATION_FAILED"

def replay_one(f,cfg):
    d=PlateauDetector(cfg); tr=[]; stop=None
    for i in range(len(f)):
        a=d.update(f,i); r=f.iloc[i]
        tr.append({"landmark_index":int(r.landmark_index),"nodes":int(r.nodes),
                   "removed_fraction":float(r.removed_fraction),"active":bool(r.active),
                   "plateau":bool(r.plateau),"action":a})
        if a=="STOP_CONFIRMED_PLATEAU":
            stop={"stop_landmark":int(r.landmark_index),"stop_nodes":int(r.nodes),
                  "stop_removed_fraction":float(r.removed_fraction),
                  "entry_landmark":int(d.entry),"chosen_regime_landmark":int(d.best or r.landmark_index)}
            break
    return pd.DataFrame(tr),stop

def nearest_robust(stop,stability):
    if stop is None: return None
    s=stability[stability.stable.astype(bool)].copy()
    if len(s)==0:return None
    s["drift"]=np.abs(s.removed_fraction-float(stop["stop_removed_fraction"]))
    r=s.sort_values(["drift","support_fraction"],ascending=[True,False]).iloc[0]
    return {"audited_nodes":int(r.nodes),"audited_removed_fraction":float(r.removed_fraction),
            "audit_support_fraction":float(r.support_fraction),"delta_removed_fraction":float(r.drift)}

def replay_grid(df,norm,stability,cfg):
    rows=[]; traces={}
    for rw in cfg["recent_widths"]:
        for hw in cfg["history_widths"]:
            if hw<=rw: continue
            f,meta=build_future_blind_features(df,norm,cfg,rw,hw)
            tr,stop=replay_one(f,cfg); near=nearest_robust(stop,stability)
            aligned=near is not None and near["delta_removed_fraction"]<=float(cfg["max_allowed_removed_fraction_drift"])
            rows.append({"recent_width":rw,"history_width":hw,"stop_found":stop is not None,
                         "stop_nodes":stop["stop_nodes"] if stop else np.nan,
                         "stop_removed_fraction":stop["stop_removed_fraction"] if stop else np.nan,
                         "audited_nodes":near["audited_nodes"] if near else np.nan,
                         "delta_removed_fraction":near["delta_removed_fraction"] if near else np.nan,
                         "aligned":bool(aligned)})
            traces[(rw,hw)]=(f,tr,stop,near,meta)
    return pd.DataFrame(rows),traces

def consensus(grid,cfg):
    n=len(grid); found=grid[grid.stop_found.astype(bool)]; aligned=grid[grid.aligned.astype(bool)]
    sf=len(found)/max(n,1); af=len(aligned)/max(n,1)
    if len(aligned):
        medr=float(np.median(aligned.stop_removed_fraction))
        medn=int(round(float(np.median(aligned.stop_nodes))))
        iqr=float(np.quantile(aligned.stop_removed_fraction,.75)-np.quantile(aligned.stop_removed_fraction,.25))
    else: medr=np.nan; medn=0; iqr=np.nan
    robust=(sf>=float(cfg["min_grid_stop_fraction"]) and af>=float(cfg["min_grid_alignment_fraction"])
            and np.isfinite(iqr) and iqr<=float(cfg["max_consensus_removed_iqr"]))
    return {"grid_configurations":n,"stop_fraction":sf,"alignment_fraction":af,
            "consensus_removed_fraction":medr,"consensus_nodes":medn,
            "consensus_removed_iqr":iqr,"robust":bool(robust)}

def detector_source_is_future_blind():
    src=inspect.getsource(build_future_blind_features)
    forbidden=("quantile(","center=True","bfill(","backfill(","limit_direction")
    return all(x not in src for x in forbidden)
