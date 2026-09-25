from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

def finite_values(x):
    a=np.asarray(x,float)
    return a[np.isfinite(a)]

def safe_median(x):
    a=finite_values(x)
    return float(np.median(a)) if len(a) else np.nan

def safe_quantile(x,q):
    a=finite_values(x)
    return float(np.quantile(a,float(q))) if len(a) else np.nan

def safe_mean(x):
    a=finite_values(x)
    return float(a.mean()) if len(a) else np.nan

def safe_std(x):
    a=finite_values(x)
    return float(a.std()) if len(a) else np.nan

def shannon_from_labels(labels):
    if labels is None or len(labels)==0:
        return np.nan
    _,cnt=np.unique(np.asarray(labels),return_counts=True)
    p=cnt/cnt.sum()
    return float(-(p*np.log(p)).sum())

def detect_basins(gates,candidates,cfg):
    """
    Build finite persistence basins around v1.0.3 local minima.

    A basin consists of contiguous landmarks around a candidate for which the
    plateau-distance score remains within a relative shoulder tolerance of the
    minimum. This is diagnostic only; it does not alter any stopping gate.
    """
    score=gates.plateau_distance_score.to_numpy(float)
    out=[]
    shoulder=float(cfg["basin_shoulder_multiplier"])
    min_width=int(cfg["min_basin_landmarks"])
    for c in candidates.itertuples(index=False):
        i=int(c.table_index)
        if i<0 or i>=len(gates) or not np.isfinite(score[i]):
            continue
        s0=float(score[i])
        thresh=max(s0*shoulder,s0+float(cfg["basin_absolute_floor"]))
        lo=i
        while lo>0 and np.isfinite(score[lo-1]) and score[lo-1]<=thresh:
            lo-=1
        hi=i
        while hi+1<len(score) and np.isfinite(score[hi+1]) and score[hi+1]<=thresh:
            hi+=1
        if hi-lo+1<min_width:
            continue

        left_shoulder=safe_median(score[max(0,lo-int(cfg["shoulder_window"])):lo])
        right_shoulder=safe_median(score[hi+1:min(len(score),hi+1+int(cfg["shoulder_window"]))])
        shoulders=finite_values([left_shoulder,right_shoulder])
        shoulder_level=float(np.median(shoulders)) if len(shoulders) else np.nan
        depth=(shoulder_level-s0) if np.isfinite(shoulder_level) else np.nan

        out.append({
            "candidate_landmark":int(c.landmark_index),
            "candidate_table_index":i,
            "entry_table_index":int(lo),
            "exit_table_index":int(hi),
            "entry_landmark":int(gates.landmark_index.iloc[lo]),
            "exit_landmark":int(gates.landmark_index.iloc[hi]),
            "entry_nodes":int(gates.nodes.iloc[lo]),
            "minimum_nodes":int(gates.nodes.iloc[i]),
            "exit_nodes":int(gates.nodes.iloc[hi]),
            "entry_removed_fraction":float(gates.removed_fraction.iloc[lo]),
            "minimum_removed_fraction":float(gates.removed_fraction.iloc[i]),
            "exit_removed_fraction":float(gates.removed_fraction.iloc[hi]),
            "entry_ell":float(gates.ell.iloc[lo]) if "ell" in gates.columns else np.nan,
            "minimum_ell":float(gates.ell.iloc[i]) if "ell" in gates.columns else np.nan,
            "exit_ell":float(gates.ell.iloc[hi]) if "ell" in gates.columns else np.nan,
            "lifetime_ell":(
                float(gates.ell.iloc[hi]-gates.ell.iloc[lo])
                if "ell" in gates.columns else np.nan
            ),
            "basin_landmarks":int(hi-lo+1),
            "minimum_score":s0,
            "shoulder_level":shoulder_level,
            "basin_depth":float(depth) if np.isfinite(depth) else np.nan,
        })
    if not out:
        return pd.DataFrame()
    b=pd.DataFrame(out)
    # deduplicate heavily overlapping basins by keeping the deeper minimum
    keep=[]
    for r in b.sort_values(["minimum_score","basin_depth"],ascending=[True,False]).itertuples(index=False):
        overlap=False
        for k in keep:
            inter=max(0,min(r.exit_table_index,k.exit_table_index)-max(r.entry_table_index,k.entry_table_index)+1)
            union=max(r.exit_table_index,k.exit_table_index)-min(r.entry_table_index,k.entry_table_index)+1
            if union>0 and inter/union>=float(cfg["basin_overlap_dedup"]):
                overlap=True; break
        if not overlap: keep.append(r)
    return pd.DataFrame([x._asdict() for x in sorted(keep,key=lambda z:z.minimum_removed_fraction)])

def attach_block_coherence(basins,block_ratios,cfg):
    if len(basins)==0:
        return pd.DataFrame(),pd.DataFrame()
    detail=[];summary=[]
    blocks=sorted(block_ratios.block.unique())
    thr=float(cfg["closure_ratio_max"])
    for bid,b in enumerate(basins.itertuples(index=False)):
        z=block_ratios[
            (block_ratios.table_index>=int(b.entry_table_index))&
            (block_ratios.table_index<=int(b.exit_table_index))
        ]
        stable_blocks=0
        active_blocks=[]
        for block in blocks:
            g=z[z.block==block]
            vals=finite_values(g.closure_ratio)
            med=float(np.median(vals)) if len(vals) else np.nan
            q75=float(np.quantile(vals,.75)) if len(vals) else np.nan
            sf=float(np.mean(vals<=thr)) if len(vals) else np.nan
            stable=bool(np.isfinite(sf) and sf>=float(cfg["block_stable_fraction_required"]))
            stable_blocks+=int(stable)
            if not stable: active_blocks.append(block)
            detail.append({
                "basin_id":bid,"candidate_landmark":int(b.candidate_landmark),
                "block":block,"resolved_values":int(len(vals)),
                "median_closure_ratio":med,"q75_closure_ratio":q75,
                "stable_fraction":sf,"stable":stable
            })
        coherence=stable_blocks/max(len(blocks),1)
        labels=[]
        for _,g in z.groupby("table_index"):
            meds=[]
            for block in blocks:
                vals=finite_values(g[g.block==block].closure_ratio)
                if len(vals):
                    meds.append((float(np.median(vals)),block))
            if meds:
                labels.append(max(meds)[1])
        entropy=shannon_from_labels(labels)
        summary.append({
            "basin_id":bid,"candidate_landmark":int(b.candidate_landmark),
            "stable_blocks":int(stable_blocks),"total_blocks":int(len(blocks)),
            "cross_block_coherence":float(coherence),
            "unstable_block_switch_entropy":entropy,
            "active_blocks":";".join(active_blocks)
        })
    return pd.DataFrame(summary),pd.DataFrame(detail)

def hierarchy_landmarks(basins,coherence,cfg):
    if len(basins)==0:return pd.DataFrame()
    x=basins.merge(coherence,on="candidate_landmark",how="left")
    lifetime=x.lifetime_ell.to_numpy(float)
    depth=x.basin_depth.to_numpy(float)
    coh=x.cross_block_coherence.to_numpy(float)
    # rank-normalized composite; only finite evidence contributes
    def rank01(a):
        out=np.full(len(a),np.nan)
        q=np.isfinite(a)
        if q.sum()==1: out[q]=1.
        elif q.sum()>1:
            r=pd.Series(a[q]).rank(method="average").to_numpy()
            out[q]=(r-1)/(len(r)-1)
        return out
    rl=rank01(lifetime); rd=rank01(depth); rc=rank01(coh)
    M=np.vstack([rl,rd,rc]).T
    score=np.full(len(x),np.nan)
    for i,row in enumerate(M):
        z=row[np.isfinite(row)]
        if len(z): score[i]=float(np.mean(z))
    x["landmark_strength"]=score
    x["hierarchy_landmark"]=(
        (x.basin_landmarks>=int(cfg["min_basin_landmarks"]))&
        (x.cross_block_coherence>=float(cfg["min_cross_block_coherence"]))&
        np.isfinite(x.landmark_strength)
    )
    x=x.sort_values("minimum_removed_fraction").reset_index(drop=True)
    x["landmark_order"]=np.arange(1,len(x)+1)
    return x

def spatial_stats_from_node_state(df):
    """
    Defensive spatial summary from a checkpoint/landmark node-state table.
    Uses only fields that actually exist.
    """
    out={}
    if df is None or len(df)==0:return out
    xcol=next((c for c in ["centroid_x","x","x_coord"] if c in df.columns),None)
    ycol=next((c for c in ["centroid_y","y","y_coord"] if c in df.columns),None)
    if xcol and ycol:
        x=df[xcol].to_numpy(float); y=df[ycol].to_numpy(float)
        q=np.isfinite(x)&np.isfinite(y)
        if q.any():
            out["spatial_centroid_x"]=float(np.mean(x[q]))
            out["spatial_centroid_y"]=float(np.mean(y[q]))
            out["spatial_extent_x"]=float(np.max(x[q])-np.min(x[q]))
            out["spatial_extent_y"]=float(np.max(y[q])-np.min(y[q]))
            out["spatial_rms_radius"]=float(np.sqrt(np.mean(
                (x[q]-np.mean(x[q]))**2+(y[q]-np.mean(y[q]))**2
            )))
    if "size" in df.columns:
        s=finite_values(df["size"])
        if len(s):
            out["supernode_size_mean"]=float(np.mean(s))
            out["supernode_size_q95"]=float(np.quantile(s,.95))
            out["supernode_size_max"]=float(np.max(s))
    return out

def gene_state_stats_from_node_state(df):
    out={}
    if df is None or len(df)==0:return out
    for c in ["expression_mean","expression_variance"]:
        if c in df.columns:
            vals=finite_values(df[c])
            if len(vals):
                out[c+"_mean"]=float(vals.mean())
                out[c+"_std"]=float(vals.std())
    return out

def find_landmark_node_state(ledger_root,sample,landmark_index):
    """
    Search existing ledger exports without assuming one hard-coded filename.
    Returns (df,path) or (None,None). Never fabricates ancestry/spatial data.
    """
    root=Path(ledger_root)
    if not root.exists(): return None,None
    pats=[
        f"**/{sample}/**/*landmark*{int(landmark_index)}*node*.parquet",
        f"**/{sample}/**/*node*{int(landmark_index)}*.parquet",
        f"**/{sample}/**/*landmark*{int(landmark_index)}*.parquet",
    ]
    seen=set()
    for pat in pats:
        for p in root.glob(pat):
            if p in seen or not p.is_file(): continue
            seen.add(p)
            try:
                d=pd.read_parquet(p)
            except Exception:
                continue
            if any(c in d.columns for c in ["supernode_id","size","centroid_x","expression_mean"]):
                return d,p
    return None,None
