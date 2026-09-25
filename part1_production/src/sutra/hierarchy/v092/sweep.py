from __future__ import annotations
import math
import numpy as np
import pandas as pd

def canonical_pair(i,j):
    i=int(i); j=int(j)
    return (i,j) if i<j else (j,i)

def aggregate_completed_edges(edges, labels):
    """Aggregate completed Level-0 backbone edges onto current supernodes.

    All fields remain descriptive; no modality becomes a hard veto.
    Edge costs are recomputed from superedge averages and confidence-weighted
    modality terms.
    """
    lab=np.asarray(labels,dtype=np.int64)
    i0=edges.cell_i_index.to_numpy(np.int64)
    j0=edges.cell_j_index.to_numpy(np.int64)
    si=lab[i0]; sj=lab[j0]
    q=si!=sj
    if not q.any():
        return pd.DataFrame()

    x=edges.loc[q].copy()
    x["super_i"]=np.minimum(si[q],sj[q])
    x["super_j"]=np.maximum(si[q],sj[q])

    num_cols=[
        "spatial_expression_weight",
        "expression_distance",
        "normalized_spatial_distance",
        "functional_distance",
        "functional_similarity",
        "cellchat_forward",
        "cellchat_reverse",
        "cellchat_total",
        "cellchat_directionality",
        "cellchat_support_forward",
        "cellchat_support_reverse",
        "tension_complete",
        "tension_confidence",
        "delta_p_potential_complete",
        "pressure_confidence",
        "spatial_distance",
        "gap_guard_ratio",
    ]
    num_cols=[c for c in num_cols if c in x.columns]

    agg={c:"mean" for c in num_cols}
    agg["cell_i_index"]="count"
    if "source_contact" in x.columns:
        agg["source_contact"]="mean"

    y=x.groupby(["super_i","super_j"],sort=False).agg(agg).reset_index()
    y=y.rename(columns={"cell_i_index":"n_level0_edges"})
    return y

def robust_scale(a):
    x=np.asarray(a,dtype=float)
    x=x[np.isfinite(x)]
    if len(x)==0:
        return 1.0
    q25,q75=np.quantile(x,[.25,.75])
    s=(q75-q25)/1.349
    if not np.isfinite(s) or s<=1e-12:
        s=np.median(np.abs(x-np.median(x)))*1.4826
    if not np.isfinite(s) or s<=1e-12:
        s=max(np.std(x),1.0)
    return float(max(s,1e-12))

def completed_pair_cost(superedges, weights):
    """Continuous multimodal cost; no missing-modality veto."""
    x=superedges.copy()

    terms={}
    # Mandatory everywhere-defined backbone terms.
    terms["space_expr"] = (
        x.expression_distance.to_numpy(float)
        + x.normalized_spatial_distance.to_numpy(float)
    )
    terms["functional"] = x.functional_distance.to_numpy(float)

    # Communication mismatch weighted by actual support.
    comm_mag=np.abs(x.cellchat_directionality.to_numpy(float))
    comm_sup=np.maximum(
        x.cellchat_support_forward.to_numpy(float),
        x.cellchat_support_reverse.to_numpy(float)
    )
    terms["communication"]=comm_mag*comm_sup

    # Mechanics/pressure influence is confidence weighted, never a universal veto.
    terms["tension"]=(
        np.abs(x.tension_complete.to_numpy(float))
        *x.tension_confidence.to_numpy(float)
    )
    terms["pressure"]=(
        np.abs(x.delta_p_potential_complete.to_numpy(float))
        *x.pressure_confidence.to_numpy(float)
    )

    cost=np.zeros(len(x),dtype=float)
    for name,a in terms.items():
        z=a/robust_scale(a)
        x[f"cost_{name}"]=z
        cost += float(weights[name])*z

    x["completed_pair_cost"]=cost
    return x

def greedy_disjoint(superedges, cap):
    if len(superedges)==0 or cap<=0:
        return superedges.iloc[:0].copy()
    x=superedges.sort_values(
        ["completed_pair_cost","super_i","super_j"],
        kind="mergesort"
    )
    used=set(); rows=[]
    for _,r in x.iterrows():
        u=int(r.super_i); v=int(r.super_j)
        if u in used or v in used:
            continue
        used.add(u); used.add(v); rows.append(r)
        if len(rows)>=cap:
            break
    return pd.DataFrame(rows,columns=x.columns)

def apply_batch(labels,batch):
    out=np.asarray(labels,dtype=np.int64).copy()
    for r in batch.itertuples(index=False):
        u,v=canonical_pair(r.super_i,r.super_j)
        out[out==v]=u
    return out

def run_schedule(edges,n_cells,fraction,weights,max_steps=20000,target_removed=None):
    labels=np.arange(n_cells,dtype=np.int64)
    history=[]
    total=0

    for step in range(max_steps):
        n_nodes=len(np.unique(labels))
        se=aggregate_completed_edges(edges,labels)
        if len(se)==0:
            stop="no_backbone_superedges"
            break

        se=completed_pair_cost(se,weights)
        cap=max(1,int(math.ceil(n_nodes*fraction)))
        cap=min(cap,n_nodes//2)
        batch=greedy_disjoint(se,cap)

        if len(batch)==0:
            stop="no_disjoint_batch"
            break

        before=n_nodes
        labels=apply_batch(labels,batch)
        after=len(np.unique(labels))
        total += len(batch)
        removed=(n_cells-after)/n_cells

        history.append({
            "step":step,
            "nodes_before":before,
            "nodes_after":after,
            "selected":len(batch),
            "removed_fraction":removed,
            "pair_cost_mean":float(batch.completed_pair_cost.mean()),
            "pair_cost_q95":float(batch.completed_pair_cost.quantile(.95)),
            "candidate_superedges":int(len(se)),
        })

        if target_removed is not None and removed>=target_removed:
            stop="target_removed_reached"
            break
    else:
        stop="safety_ceiling"

    return labels,pd.DataFrame(history),{
        "fraction":float(fraction),
        "steps":int(len(history)),
        "merges":int(total),
        "final_nodes":int(len(np.unique(labels))),
        "removed_fraction":float((n_cells-len(np.unique(labels)))/n_cells),
        "stop_reason":stop,
    }

def partition_pair_agreement(a,b):
    """Rand-style agreement sampled exactly through contingency counts."""
    a=np.asarray(a); b=np.asarray(b)
    n=len(a)
    if n<2: return 1.0

    def comb2(x): return x*(x-1)//2
    _,ca=np.unique(a,return_counts=True)
    _,cb=np.unique(b,return_counts=True)
    pairs_a=sum(comb2(int(x)) for x in ca)
    pairs_b=sum(comb2(int(x)) for x in cb)

    df=pd.DataFrame({"a":a,"b":b})
    cc=df.groupby(["a","b"]).size().to_numpy()
    both=sum(comb2(int(x)) for x in cc)
    total=comb2(n)
    neither=total-pairs_a-pairs_b+both
    return float((both+neither)/total)

def label_overlap_jaccard(a,b):
    """Mean best Jaccard overlap of blocks, symmetric."""
    a=np.asarray(a); b=np.asarray(b)
    def one(x,y):
        vals=np.unique(x); scores=[]
        for v in vals:
            ix=np.flatnonzero(x==v)
            ys,counts=np.unique(y[ix],return_counts=True)
            k=np.argmax(counts); w=ys[k]
            inter=counts[k]
            union=len(ix)+np.sum(y==w)-inter
            scores.append(inter/union)
        return float(np.mean(scores)) if scores else 1.0
    return .5*(one(a,b)+one(b,a))

def interpolate_history(hist, grid):
    if len(hist)==0:
        return pd.DataFrame({"removed_fraction":grid})
    x=hist.removed_fraction.to_numpy(float)
    out={"removed_fraction":grid}
    for c in ["pair_cost_mean","pair_cost_q95","candidate_superedges","nodes_after"]:
        y=hist[c].to_numpy(float)
        out[c]=np.interp(grid,x,y,left=y[0],right=y[-1])
    return pd.DataFrame(out)
