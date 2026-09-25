from __future__ import annotations
import numpy as np
import pandas as pd

BLOCK_COLUMNS={
    "expression":[
        "expression_total_variance",
        "dominant_expression_coherence",
        "gene_mean_across_supernodes_q50",
        "gene_variance_across_supernodes_q50",
    ],
    "go_msigdb":[
        "functional_distance_mean","functional_distance_q50","functional_distance_q95",
        "functional_similarity_mean",
    ],
    "cellchat":[
        "cellchat_total_mean","cellchat_total_q50","cellchat_total_q95",
        "cellchat_directionality_mean","cellchat_directionality_q50",
        "cellchat_support_forward_mean","cellchat_support_reverse_mean",
    ],
    "mechanics":[
        "tension_complete_mean","tension_complete_q50","tension_complete_q95",
        "tension_confidence_mean","tension_confidence_q50",
        "tension_anchor_supported_fraction",
        "tension_harmonic_extension_fraction",
        "tension_neutral_prior_fraction",
    ],
    "pressure":[
        "pressure_difference_mean","pressure_difference_q50","pressure_difference_q95",
        "pressure_confidence_mean","pressure_confidence_q50",
    ],
    "topology":[
        "components","cycle_rank","mean_degree","max_degree",
    ],
    "local_geometry":[
        "spatial_distance_mean","spatial_distance_q50","spatial_distance_q95",
        "gap_guard_ratio_mean","gap_guard_ratio_q50","gap_guard_ratio_q95",
        "dominant_spatial_coherence",
    ],
    "directional_transport":[
        "directional_dual_norm_q50","directional_dual_norm_q95",
        "transport_rotate_fraction","transport_identity_fraction",
        "transport_unresolved_fraction",
    ],
    "geodesics":[
        "geodesic_length_q50","geodesic_length_q95",
        "geodesic_reversal_ratio_q50","geodesic_reversal_ratio_q95",
    ],
    "holonomy":[
        "holonomy_angle_rms_q50","holonomy_angle_rms_q95",
        "triangle_resolved_fraction",
    ],
    "holonomy_density":[
        "holonomy_density_q50","holonomy_density_q95","holonomy_density_mean",
    ],
    "component_structure":[
        "components","supernode_size_entropy","effective_domain_number",
        "K50","K80","K90","mass_gini",
    ],
}

def mass_statistics(sizes):
    x=np.asarray(sizes,dtype=np.int64)
    if len(x)==0 or x.sum()<=0:
        return {}
    xs=np.sort(x)[::-1]
    p=xs/xs.sum()
    cs=np.cumsum(p)
    def k(frac):
        return int(np.searchsorted(cs,float(frac),side="left")+1)
    neff=float(1.0/np.sum(p*p))
    # Standard finite-sample Gini for positive masses.
    y=np.sort(x.astype(float))
    n=len(y)
    g=float((2*np.sum((np.arange(1,n+1))*y)/(n*np.sum(y)))-((n+1)/n))
    return {
        "effective_domain_number":neff,
        "K50":k(.50),"K80":k(.80),"K90":k(.90),
        "mass_gini":g,
    }

def dominant_indices(sizes,fraction=.80):
    x=np.asarray(sizes,dtype=np.int64)
    order=np.argsort(x)[::-1]
    if len(order)==0:return np.array([],dtype=np.int64)
    cs=np.cumsum(x[order])/x.sum()
    k=int(np.searchsorted(cs,float(fraction),side="left")+1)
    return order[:k]

def node_coherence(labels,node_ids,sizes,Ynode,xy_node,cell_expr_sqmean,cell_r2):
    labels=np.asarray(labels,np.int64)
    node_ids=np.asarray(node_ids,np.int64)
    sizes=np.asarray(sizes,np.int64)

    sum_expr_sq=np.bincount(
        labels,weights=np.asarray(cell_expr_sqmean,float),
        minlength=len(labels)
    )[node_ids]
    cluster_expr_sq=sum_expr_sq/np.maximum(sizes,1)
    mean_sq=np.mean(np.asarray(Ynode,float)**2,axis=1)
    expr_mse=np.maximum(cluster_expr_sq-mean_sq,0.0)

    sum_r2=np.bincount(
        labels,weights=np.asarray(cell_r2,float),
        minlength=len(labels)
    )[node_ids]
    spatial_var=np.maximum(
        sum_r2/np.maximum(sizes,1)-np.sum(np.asarray(xy_node,float)**2,axis=1),
        0.0
    )
    spatial_rms=np.sqrt(spatial_var)

    dom=dominant_indices(sizes,.80)
    if len(dom):
        w=sizes[dom].astype(float);w/=w.sum()
        dexpr=float(np.sum(w*expr_mse[dom]))
        dspatial=float(np.sum(w*spatial_rms[dom]))
    else:
        dexpr=np.nan;dspatial=np.nan

    return (
        expr_mse.astype(float),
        spatial_rms.astype(float),
        dexpr,dspatial,
        np.isin(np.arange(len(node_ids)),dom)
    )

def symmetric_relative_change(a,b,eps=1e-12):
    a=float(a);b=float(b)
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return float(2.0*abs(b-a)/max(abs(a)+abs(b),eps))

def block_speed(previous,current,columns):
    vals=[]
    for c in columns:
        if c not in previous or c not in current:
            continue
        z=symmetric_relative_change(previous[c],current[c])
        if np.isfinite(z):vals.append(z)
    if not vals:return np.nan
    return float(np.sqrt(np.mean(np.square(vals))))

def attach_block_speeds(previous,current):
    out=dict(current)
    for block,cols in BLOCK_COLUMNS.items():
        out[f"speed_{block}"]=(
            np.nan if previous is None else block_speed(previous,current,cols)
        )
    return out

def add_derivatives(df,xcol="removed_fraction"):
    if len(df)==0:return df.copy()
    x=df[xcol].to_numpy(float)
    blocks=[]
    numeric=[
        c for c in df.columns
        if c!=xcol and pd.api.types.is_numeric_dtype(df[c])
    ]
    for c in numeric:
        y=df[c].to_numpy(float)
        finite=np.isfinite(y)
        if finite.sum()<2:
            continue
        yy=pd.Series(y).interpolate(limit_direction="both").to_numpy(float)
        d1=np.full(len(df),np.nan,float)
        d2=np.full(len(df),np.nan,float)
        if len(df)>=2:
            # np.gradient with x directly handles nonuniform evaluation spacing.
            try:
                d1=np.gradient(yy,x)
                d2=np.gradient(d1,x)
            except Exception:
                pass
        blocks.append(pd.DataFrame({f"d1_{c}":d1,f"d2_{c}":d2}))
    return pd.concat([df.reset_index(drop=True)]+blocks,axis=1) if blocks else df.copy()

def json_safe(x):
    if isinstance(x,dict):return {str(k):json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [json_safe(v) for v in x]
    if isinstance(x,pd.DataFrame):return x.to_dict("records")
    if isinstance(x,pd.Series):return x.to_dict()
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,)):return None if not np.isfinite(x) else float(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
