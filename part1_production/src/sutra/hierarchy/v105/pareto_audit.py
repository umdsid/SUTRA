from __future__ import annotations
import numpy as np
import pandas as pd

def finite(x):
    a=np.asarray(x,float)
    return a[np.isfinite(a)]

def safe_min(x):
    a=finite(x); return float(np.min(a)) if len(a) else np.nan
def safe_max(x):
    a=finite(x); return float(np.max(a)) if len(a) else np.nan
def safe_median(x):
    a=finite(x); return float(np.median(a)) if len(a) else np.nan
def safe_quantile(x,q):
    a=finite(x); return float(np.quantile(a,float(q))) if len(a) else np.nan

def audit_v104_binary_failures(landmarks,cfg):
    """
    Reproduce the v1.0.4 binary landmark semantics exactly enough to explain
    acceptance/rejection. No criterion is relaxed here.
    """
    x=landmarks.copy()
    min_width=int(cfg["v104_min_basin_landmarks"])
    min_coh=float(cfg["v104_min_cross_block_coherence"])

    x["pass_min_basin_landmarks"]=x["basin_landmarks"].astype(float)>=min_width
    x["pass_finite_lifetime"]=np.isfinite(x["lifetime_ell"].astype(float))
    x["pass_finite_depth"]=np.isfinite(x["basin_depth"].astype(float))
    x["pass_cross_block_coherence"]=(
        np.isfinite(x["cross_block_coherence"].astype(float))&
        (x["cross_block_coherence"].astype(float)>=min_coh)
    )
    x["pass_finite_strength"]=np.isfinite(x["landmark_strength"].astype(float))

    checks=[
        "pass_min_basin_landmarks","pass_finite_lifetime","pass_finite_depth",
        "pass_cross_block_coherence","pass_finite_strength"
    ]
    x["binary_pass_count"]=x[checks].astype(int).sum(axis=1)
    x["binary_fail_count"]=len(checks)-x["binary_pass_count"]

    def reasons(r):
        out=[]
        mapping={
            "pass_min_basin_landmarks":"minimum_basin_landmarks",
            "pass_finite_lifetime":"finite_lifetime",
            "pass_finite_depth":"finite_basin_depth",
            "pass_cross_block_coherence":"cross_block_coherence",
            "pass_finite_strength":"finite_landmark_strength",
        }
        for c,name in mapping.items():
            if not bool(r[c]): out.append(name)
        return ";".join(out) if out else "none"

    x["binary_failure_reasons"]=x.apply(reasons,axis=1)

    # Check that the reconstructed decision agrees with the stored v1.0.4 flag.
    reproduced=(
        x["pass_min_basin_landmarks"]&
        x["pass_cross_block_coherence"]&
        x["pass_finite_strength"]
    )
    x["v104_binary_reproduced"]=reproduced.astype(bool)
    x["v104_flag_agrees"]=(
        x["v104_binary_reproduced"].astype(bool)==
        x["hierarchy_landmark"].astype(bool)
    )
    return x

def criterion_summary(audited):
    rows=[]
    criteria=[
        ("minimum_basin_landmarks","pass_min_basin_landmarks"),
        ("finite_lifetime","pass_finite_lifetime"),
        ("finite_basin_depth","pass_finite_depth"),
        ("cross_block_coherence","pass_cross_block_coherence"),
        ("finite_landmark_strength","pass_finite_strength"),
    ]
    for name,col in criteria:
        rows.append({
            "criterion":name,
            "basins":int(len(audited)),
            "pass_count":int(audited[col].sum()),
            "fail_count":int((~audited[col].astype(bool)).sum()),
            "pass_fraction":float(audited[col].mean()) if len(audited) else np.nan
        })
    return pd.DataFrame(rows)

def coherence_distribution(audited):
    c=finite(audited.cross_block_coherence)
    if not len(c):
        return {
            "n":0,"min":np.nan,"q10":np.nan,"q25":np.nan,"median":np.nan,
            "q75":np.nan,"q90":np.nan,"max":np.nan
        }
    return {
        "n":int(len(c)),"min":float(np.min(c)),
        "q10":float(np.quantile(c,.10)),"q25":float(np.quantile(c,.25)),
        "median":float(np.median(c)),"q75":float(np.quantile(c,.75)),
        "q90":float(np.quantile(c,.90)),"max":float(np.max(c))
    }

def threshold_sensitivity(audited,thresholds):
    """
    Diagnostic only: count how many already-observed basins would survive each
    coherence cutoff. No cutoff is selected by this routine.
    """
    rows=[]
    base=(
        audited.pass_min_basin_landmarks.astype(bool)&
        audited.pass_finite_strength.astype(bool)
    )
    c=audited.cross_block_coherence.to_numpy(float)
    for t in thresholds:
        q=base & np.isfinite(c) & (c>=float(t))
        rows.append({
            "coherence_threshold":float(t),
            "surviving_basins":int(q.sum()),
            "surviving_fraction":float(q.mean()) if len(q) else np.nan
        })
    return pd.DataFrame(rows)

def pareto_front(df,metrics):
    """
    Maximize all supplied metrics. Missing values make a basin ineligible for
    this specific Pareto front rather than being imputed.
    """
    x=df.copy()
    eligible=np.ones(len(x),dtype=bool)
    A=[]
    for m in metrics:
        a=x[m].to_numpy(float)
        eligible &= np.isfinite(a)
        A.append(a)
    M=np.vstack(A).T if A else np.empty((len(x),0))
    front=np.zeros(len(x),dtype=bool)
    dominated_by=np.full(len(x),-1,int)

    idx=np.where(eligible)[0]
    for i in idx:
        dominated=False
        for j in idx:
            if i==j: continue
            weak=np.all(M[j]>=M[i])
            strict=np.any(M[j]>M[i])
            if weak and strict:
                dominated=True
                dominated_by[i]=j
                break
        front[i]=not dominated

    x["pareto_eligible"]=eligible
    x["pareto_front"]=front
    x["dominated_by_row"]=dominated_by
    return x

def pareto_layers(df,metrics,max_layers=8):
    """
    Repeated non-dominated sorting. Layer 1 is the primary Pareto front.
    """
    work=df.copy()
    work["pareto_layer"]=np.nan
    remaining=np.arange(len(work))
    layer=1
    while len(remaining) and layer<=int(max_layers):
        sub=pareto_front(work.iloc[remaining].reset_index(),metrics)
        local=np.where(sub.pareto_front.to_numpy(bool))[0]
        if not len(local): break
        original=remaining[local]
        work.loc[original,"pareto_layer"]=layer
        mask=np.ones(len(remaining),dtype=bool)
        mask[local]=False
        remaining=remaining[mask]
        layer+=1
    return work

def nearest_scale_spacing(audited):
    x=audited.sort_values("minimum_removed_fraction").copy()
    r=x.minimum_removed_fraction.to_numpy(float)
    prev=np.full(len(x),np.nan); nxt=np.full(len(x),np.nan)
    if len(x)>1:
        prev[1:]=r[1:]-r[:-1]
        nxt[:-1]=r[1:]-r[:-1]
    x["spacing_from_previous_removed_fraction"]=prev
    x["spacing_to_next_removed_fraction"]=nxt
    return x

def attach_state_descriptors(audited,state):
    """
    Attach the v1.0.4 node-state descriptors by candidate landmark when present.
    Missing state descriptors remain missing.
    """
    if state is None or len(state)==0:
        return audited.copy()
    if "candidate_landmark" not in state.columns:
        return audited.copy()
    cols=["candidate_landmark"]+[c for c in state.columns if c!="candidate_landmark"]
    return audited.merge(state[cols],on="candidate_landmark",how="left",suffixes=("","_state"))

def block_failure_profile(block_detail):
    """
    For each basin, identify stable/unstable block counts and the least coherent
    blocks using the already-computed v1.0.4 block detail.
    """
    if block_detail is None or len(block_detail)==0:
        return pd.DataFrame()
    rows=[]
    for bid,g in block_detail.groupby("basin_id"):
        stable=int(g.stable.astype(bool).sum()) if "stable" in g.columns else 0
        total=int(len(g))
        ranked=g.copy()
        ranked["rank_unstable"]=ranked["median_closure_ratio"].rank(
            method="dense",ascending=False,na_option="bottom"
        )
        top=ranked[ranked.rank_unstable==1]
        rows.append({
            "basin_id":int(bid),
            "stable_blocks":stable,
            "total_blocks":total,
            "stable_block_fraction":float(stable/max(total,1)),
            "least_coherent_blocks":";".join(sorted(top.block.astype(str).tolist())),
            "max_median_closure_ratio":safe_max(ranked.median_closure_ratio)
        })
    return pd.DataFrame(rows)

def pareto_stability_leave_one_metric_out(audited,metrics):
    """
    Audit whether front membership is entirely driven by one metric.
    """
    base=pareto_front(audited,metrics)
    base_ids=set(base.loc[base.pareto_front,"candidate_landmark"].astype(int))
    rows=[]
    for drop in metrics:
        mm=[m for m in metrics if m!=drop]
        q=pareto_front(audited,mm)
        ids=set(q.loc[q.pareto_front,"candidate_landmark"].astype(int))
        inter=len(base_ids&ids); union=len(base_ids|ids)
        rows.append({
            "dropped_metric":drop,
            "remaining_metrics":";".join(mm),
            "base_front_size":len(base_ids),
            "loo_front_size":len(ids),
            "front_jaccard":float(inter/union) if union else 1.0
        })
    return pd.DataFrame(rows)
