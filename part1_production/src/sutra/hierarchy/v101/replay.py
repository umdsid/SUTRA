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
    for b,prefixes in BLOCKS.items():
        if any(str(c).startswith(p) for p in prefixes):
            return b
    return None

def finite_rms(A,axis=1):
    """NaN-preserving RMS with no empty-slice warning."""
    A=np.asarray(A,float)
    finite=np.isfinite(A)
    count=finite.sum(axis=axis)
    sq=np.where(finite,A*A,0.0).sum(axis=axis)
    out=np.full(np.shape(count),np.nan,float)
    q=count>0
    out[q]=np.sqrt(sq[q]/count[q])
    return out

def prepare_frozen_coordinates(df,norm,min_finite_fraction=.65):
    """Use frozen audit normalization, but never impute missing landmark values."""
    cols=[];blocks=[];Z=[]
    for r in norm.itertuples(index=False):
        c=str(r.observable)
        b=block_for(c)
        if b is None or c not in df.columns:
            continue
        x=df[c].to_numpy(float)
        if np.isfinite(x).mean()<min_finite_fraction:
            continue
        s=float(r.robust_scale)
        if not np.isfinite(s) or s<=1e-12:
            continue
        z=(x-float(r.median))/s
        cols.append(c);blocks.append(b);Z.append(z)
    if not Z:
        return [],[],np.empty((len(df),0))
    return cols,blocks,np.stack(Z,axis=1)

def causal_difference(Y,x,stride):
    """Backward finite difference only. Missing endpoints stay unresolved."""
    Y=np.asarray(Y,float); x=np.asarray(x,float)
    out=np.full_like(Y,np.nan,float)
    s=max(1,int(stride))
    for i in range(s,len(x)):
        dx=float(x[i]-x[i-s])
        if not np.isfinite(dx) or abs(dx)<=1e-12:
            continue
        q=np.isfinite(Y[i])&np.isfinite(Y[i-s])
        out[i,q]=(Y[i,q]-Y[i-s,q])/dx
    return out

def trailing_nanmedian_causal(x,width):
    """At index i, use only [i-width+1, i]. Never centered."""
    x=np.asarray(x,float)
    out=np.full(len(x),np.nan)
    w=max(1,int(width))
    for i in range(len(x)):
        z=x[max(0,i-w+1):i+1]
        z=z[np.isfinite(z)]
        if len(z):
            out[i]=float(np.median(z))
    return out

def block_norm_matrix(D,blocks,expected_blocks):
    """Return per-block RMS and number of represented blocks at each landmark."""
    n=D.shape[0]
    B=np.full((n,len(expected_blocks)),np.nan,float)
    for k,b in enumerate(expected_blocks):
        idx=[j for j,x in enumerate(blocks) if x==b]
        if idx:
            B[:,k]=finite_rms(D[:,idx],axis=1)
    resolved=np.isfinite(B).sum(axis=1)
    balanced=finite_rms(B,axis=1)
    return B,resolved,balanced

def exact_causal_configs(df,norm,cfg):
    """
    Compute the exact causal estimator over the exhaustive trajectory.

    A configuration is defined by derivative stride and trailing smoothing width.
    Velocity and acceleration are equal-block RMS values. A configuration is
    unresolved unless both quantities exist and enough scientific blocks are
    represented.
    """
    cols,blocks,Z=prepare_frozen_coordinates(
        df,norm,float(cfg["min_finite_fraction"])
    )
    expected=sorted(set(blocks))
    nblocks=len(expected)
    min_blocks=max(
        int(cfg["min_resolved_blocks_absolute"]),
        int(math.ceil(float(cfg["min_resolved_block_fraction"])*nblocks))
    )
    ell=df.ell.to_numpy(float)
    rows=[]
    block_rows=[]

    for stride in cfg["derivative_strides"]:
        V=causal_difference(Z,ell,int(stride))
        A=causal_difference(V,ell,int(stride))
        BV,rv,vraw=block_norm_matrix(V,blocks,expected)
        BA,ra,araw=block_norm_matrix(A,blocks,expected)
        rblocks=np.minimum(rv,ra)

        for width in cfg["causal_smoothing_windows"]:
            vs=trailing_nanmedian_causal(vraw,int(width))
            ac=trailing_nanmedian_causal(araw,int(width))
            for i in range(len(df)):
                resolved=(
                    np.isfinite(vs[i]) and np.isfinite(ac[i])
                    and int(rblocks[i])>=min_blocks
                )
                rows.append({
                    "landmark_index":int(df.landmark_index.iloc[i]),
                    "table_index":int(i),
                    "nodes":int(df.nodes.iloc[i]),
                    "removed_fraction":float(df.removed_fraction.iloc[i]),
                    "ell":float(df.ell.iloc[i]),
                    "stride":int(stride),
                    "width":int(width),
                    "velocity":float(vs[i]) if np.isfinite(vs[i]) else np.nan,
                    "acceleration":float(ac[i]) if np.isfinite(ac[i]) else np.nan,
                    "resolved_blocks":int(rblocks[i]),
                    "expected_blocks":int(nblocks),
                    "resolved":bool(resolved),
                })

        # Export block resolution/contribution for unsmoothed derivatives.
        for i in range(len(df)):
            for k,b in enumerate(expected):
                block_rows.append({
                    "landmark_index":int(df.landmark_index.iloc[i]),
                    "table_index":int(i),
                    "stride":int(stride),
                    "block":b,
                    "velocity_block_rms":float(BV[i,k]) if np.isfinite(BV[i,k]) else np.nan,
                    "acceleration_block_rms":float(BA[i,k]) if np.isfinite(BA[i,k]) else np.nan,
                    "velocity_block_resolved":bool(np.isfinite(BV[i,k])),
                    "acceleration_block_resolved":bool(np.isfinite(BA[i,k])),
                })
    return pd.DataFrame(rows),pd.DataFrame(block_rows),{
        "observables":len(cols),
        "expected_blocks":expected,
        "n_expected_blocks":nblocks,
        "min_resolved_blocks":min_blocks,
    }

def calibrate_per_config(config_df,df,cfg):
    """Thresholds come from the same exact causal estimator, per configuration."""
    eligible_landmarks=(
        (df.nodes.to_numpy(float)>=int(cfg["min_effective_nodes"]))
        &(df.removed_fraction.to_numpy(float)>=float(cfg["calibration_min_removed_fraction"]))
        &(df.removed_fraction.to_numpy(float)<=float(cfg["calibration_max_removed_fraction"]))
    )
    elig=set(df.loc[eligible_landmarks,"landmark_index"].astype(int))
    rows=[]
    for (s,w),g in config_df.groupby(["stride","width"]):
        q=g.resolved & g.landmark_index.isin(elig)
        z=g[q]
        if len(z)<int(cfg["min_calibration_points_per_config"]):
            rows.append({
                "stride":int(s),"width":int(w),"calibrated":False,
                "n_calibration":int(len(z)),
                "velocity_low":np.nan,"acceleration_low":np.nan,
                "velocity_high":np.nan,"acceleration_high":np.nan,
            })
            continue
        rows.append({
            "stride":int(s),"width":int(w),"calibrated":True,
            "n_calibration":int(len(z)),
            "velocity_low":float(np.quantile(z.velocity,float(cfg["velocity_low_quantile"]))),
            "acceleration_low":float(np.quantile(z.acceleration,float(cfg["acceleration_low_quantile"]))),
            "velocity_high":float(np.quantile(z.velocity,float(cfg["velocity_high_quantile"]))),
            "acceleration_high":float(np.quantile(z.acceleration,float(cfg["acceleration_high_quantile"]))),
        })
    return pd.DataFrame(rows)

def classify_landmarks(config_df,thresholds,df,cfg):
    th={(int(r.stride),int(r.width)):r for r in thresholds.itertuples(index=False) if r.calibrated}
    nominal=len(th)
    rows=[]
    for lm,g in config_df.groupby("landmark_index",sort=True):
        votes=[]
        for r in g.itertuples(index=False):
            key=(int(r.stride),int(r.width))
            if key not in th or not bool(r.resolved):
                continue
            t=th[key]
            low=(float(r.velocity)<=float(t.velocity_low)
                 and float(r.acceleration)<=float(t.acceleration_low))
            high=(float(r.velocity)>=float(t.velocity_high)
                  or float(r.acceleration)>=float(t.acceleration_high))
            votes.append((low,high,float(r.velocity),float(r.acceleration)))
        nres=len(votes)
        resolved_fraction=(nres/nominal) if nominal else 0.
        eligible_vote=(
            nres>=int(cfg["min_resolved_configs_absolute"])
            and resolved_fraction>=float(cfg["min_resolved_config_fraction"])
        )
        if votes:
            low_support=sum(v[0] for v in votes)/nres
            high_support=sum(v[1] for v in votes)/nres
            mv=float(np.median([v[2] for v in votes]))
            ma=float(np.median([v[3] for v in votes]))
        else:
            low_support=high_support=0.;mv=ma=np.nan
        if not eligible_vote:
            state="UNRESOLVED"
        elif high_support>=float(cfg["high_support_required"]):
            state="FAST_CHANGE"
        elif low_support>=float(cfg["stable_support_required"]):
            state="LOW_FLOW"
        else:
            state="REGULAR"
        base=g.iloc[0]
        rows.append({
            "landmark_index":int(lm),
            "nodes":int(base.nodes),
            "removed_fraction":float(base.removed_fraction),
            "ell":float(base.ell),
            "resolved_configs":int(nres),
            "nominal_calibrated_configs":int(nominal),
            "resolved_config_fraction":float(resolved_fraction),
            "eligible_vote":bool(eligible_vote),
            "stable_support":float(low_support),
            "high_support":float(high_support),
            "median_velocity":mv,
            "median_acceleration":ma,
            "state":state,
        })
    return pd.DataFrame(rows)

class ReplayDetector:
    def __init__(self,cfg):
        self.cfg=cfg
        self.low_run=0
        self.confirm_run=0
        self.confirming=False
        self.entry=None
        self.best=None
        self.best_v=np.inf
        self.events=[]

    def update(self,row):
        eligible=(
            int(row.nodes)>=int(self.cfg["min_effective_nodes"])
            and float(row.removed_fraction)>=float(self.cfg["detector_min_removed_fraction"])
            and float(row.removed_fraction)<=float(self.cfg["detector_max_removed_fraction"])
            and bool(row.eligible_vote)
        )
        stable=eligible and float(row.stable_support)>=float(self.cfg["stable_support_required"])
        idx=int(row.landmark_index)

        if not self.confirming:
            self.low_run=self.low_run+1 if stable else 0
            if stable and np.isfinite(row.median_velocity) and float(row.median_velocity)<self.best_v:
                self.best_v=float(row.median_velocity);self.best=idx
            if self.low_run>=int(self.cfg["persistence_landmarks"]):
                self.confirming=True
                self.entry=idx-self.low_run+1
                self.confirm_run=0
                self.events.append((idx,"ENTER_CONFIRMATION"))
                return "ENTER_CONFIRMATION"
            return "CONTINUE"

        if stable:
            self.confirm_run+=1
            if np.isfinite(row.median_velocity) and float(row.median_velocity)<self.best_v:
                self.best_v=float(row.median_velocity);self.best=idx
            if self.confirm_run>=int(self.cfg["confirmation_landmarks"]):
                self.events.append((idx,"STOP_CONFIRMED_REGIME"))
                return "STOP_CONFIRMED_REGIME"
            return "CONFIRMING"

        self.events.append((idx,"CONFIRMATION_FAILED"))
        self.confirming=False;self.low_run=0;self.confirm_run=0
        self.entry=None;self.best=None;self.best_v=np.inf
        return "CONFIRMATION_FAILED"

def replay_detector(classified,cfg):
    d=ReplayDetector(cfg)
    trace=[]
    stop=None
    for r in classified.itertuples(index=False):
        action=d.update(r)
        trace.append({
            "landmark_index":int(r.landmark_index),
            "nodes":int(r.nodes),
            "removed_fraction":float(r.removed_fraction),
            "ell":float(r.ell),
            "state":str(r.state),
            "eligible_vote":bool(r.eligible_vote),
            "resolved_config_fraction":float(r.resolved_config_fraction),
            "stable_support":float(r.stable_support),
            "high_support":float(r.high_support),
            "low_run":int(d.low_run),
            "confirm_run":int(d.confirm_run),
            "confirming":bool(d.confirming),
            "action":action,
        })
        if action=="STOP_CONFIRMED_REGIME":
            stop={
                "stop_landmark":int(r.landmark_index),
                "stop_nodes":int(r.nodes),
                "stop_removed_fraction":float(r.removed_fraction),
                "chosen_regime_landmark":int(d.best if d.best is not None else r.landmark_index),
                "entry_landmark":int(d.entry if d.entry is not None else r.landmark_index),
            }
            break
    return pd.DataFrame(trace),stop

def nearest_robust_scale(stop,stability):
    stable=stability[stability.stable.astype(bool)].copy()
    if stop is None or len(stable)==0:
        return None
    target=float(stop["stop_removed_fraction"])
    stable["delta_removed"]=np.abs(stable.removed_fraction-target)
    r=stable.sort_values(["delta_removed","support_fraction"],ascending=[True,False]).iloc[0]
    return {
        "audited_nodes":int(r.nodes),
        "audited_removed_fraction":float(r.removed_fraction),
        "audit_support_fraction":float(r.support_fraction),
        "delta_removed_fraction":float(r.delta_removed),
        "node_ratio_replay_to_audit":float(stop["stop_nodes"]/max(int(r.nodes),1)),
    }
