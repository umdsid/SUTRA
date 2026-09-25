# AUTO-MATERIALIZED FROM effective_state.py.pre_v0741
# Historical v0.7.4 semantics. Do not edit in place.

"""Effective-state propagation and statistical diagnostics for STRATA v0.7.4.

The Level-0 normalization constants remain frozen.  What changes with hierarchy
level is the state carried by each effective supernode.

Blocks propagated:
  1. functional molecular state;
  2. primitive mechanics;
  3. directed communication.

Support:
  measured Level-0 contact topology only.

Missing mechanics remains missing and is carried by explicit support fractions.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FlowConfig:
    max_levels: int = 250
    checkpoint_every: int = 2
    min_mechanics_support_fraction: float = 0.50
    eps: float = 1e-12


def aggregate_supernode_expression(
    Y: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Mean normalized expression per current supernode.

    Returns:
      node_ids, state_matrix, sizes
    """
    ids,inv,sizes=np.unique(labels,return_inverse=True,return_counts=True)
    G=Y.shape[1]
    S=np.zeros((len(ids),G),dtype=np.float64)
    np.add.at(S,inv,Y.astype(np.float64))
    S/=sizes[:,None]
    return ids.astype(np.int64),S.astype(np.float32),sizes.astype(np.int64)


def aggregate_supernode_centroids(
    xy: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray,np.ndarray]:
    ids,inv,sizes=np.unique(labels,return_inverse=True,return_counts=True)
    C=np.zeros((len(ids),2),dtype=np.float64)
    np.add.at(C,inv,xy.astype(np.float64))
    C/=sizes[:,None]
    return ids.astype(np.int64),C


def aggregate_supernode_radii(
    xy: np.ndarray,
    labels: np.ndarray,
    node_ids: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    idx={int(x):k for k,x in enumerate(node_ids)}
    r=np.zeros(len(node_ids),dtype=np.float64)
    for node in node_ids:
        k=idx[int(node)]
        q=np.flatnonzero(labels==node)
        if len(q):
            d=np.sqrt(((xy[q]-centroids[k])**2).sum(axis=1))
            r[k]=float(np.max(d))
    return r


def functional_distance_between_states(
    A: np.ndarray,
    B: np.ndarray,
    L: np.ndarray,
    lam: float,
) -> np.ndarray:
    D=A.astype(np.float64)-B.astype(np.float64)
    ident=np.einsum("ij,ij->i",D,D,optimize=True)
    if lam>0 and np.any(L):
        LD=D@L
        prior=np.einsum("ij,ij->i",D,LD,optimize=True)
    else:
        prior=np.zeros(len(D),dtype=np.float64)
    return np.sqrt(np.maximum(0.0,ident+lam*prior))


def supernode_boundary_table(
    edge_rel: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Map original Level-0 interfaces to current inter-supernode boundaries.
    """
    u=labels[edge_rel.cell_i_index.to_numpy(np.int64)]
    v=labels[edge_rel.cell_j_index.to_numpy(np.int64)]
    lo=np.minimum(u,v); hi=np.maximum(u,v)
    keep=lo!=hi
    if not np.any(keep):
        return pd.DataFrame()

    B=edge_rel.loc[keep].copy()
    B["super_i"]=lo[keep]
    B["super_j"]=hi[keep]
    return B


def current_boundary_relations(
    edge_rel: pd.DataFrame,
    labels: np.ndarray,
    node_ids: np.ndarray,
    states: np.ndarray,
    L: np.ndarray,
    lam: float,
) -> pd.DataFrame:
    """
    Recompute molecular state distance between effective supernodes while
    aggregating primitive mechanics and directed communication from supporting
    Level-0 interfaces.

    Molecular state is genuinely propagated; mechanics and communication remain
    support-weighted boundary observables over measured Level-0 interfaces.
    """
    B=supernode_boundary_table(edge_rel,labels)
    if len(B)==0:
        return pd.DataFrame()

    idx={int(x):k for k,x in enumerate(node_ids)}
    rows=[]
    for (a,b),g in B.groupby(["super_i","super_j"],sort=False):
        ia=idx[int(a)]; ib=idx[int(b)]
        dm=functional_distance_between_states(
            states[[ia]],states[[ib]],L,lam
        )[0]

        mech_complete=(
            g.mechanics_tension_valid.astype(bool)
            & g.mechanics_delta_p_valid.astype(bool)
        )
        gm=g[mech_complete]

        rows.append({
            "super_i":int(a),
            "super_j":int(b),
            "n_boundary_edges":int(len(g)),
            "molecular_distance":float(dm),
            "molecular_z":float(np.nanmedian(g.molecular_z))
                if "molecular_z" in g else np.nan,
            "mechanics_support_fraction":float(mech_complete.mean()),
            "abs_tension_z":(
                float(np.nanmedian(np.abs(gm.tension_z)))
                if len(gm) else np.nan
            ),
            "abs_delta_p_z":(
                float(np.nanmedian(np.abs(gm.delta_pressure_z)))
                if len(gm) else np.nan
            ),
            "tension_z_iqr":(
                float(np.nanquantile(gm.tension_z,.75)-np.nanquantile(gm.tension_z,.25))
                if len(gm)>1 else np.nan
            ),
            "delta_p_z_iqr":(
                float(np.nanquantile(gm.delta_pressure_z,.75)-np.nanquantile(gm.delta_pressure_z,.25))
                if len(gm)>1 else np.nan
            ),
            "comm_strength":float(np.nanmedian(g.comm_strength)),
            "comm_reciprocity":float(np.nanmedian(g.comm_reciprocity)),
            "comm_asymmetry":float(np.nanmedian(
                np.abs(g.comm_ij-g.comm_ji)/(g.comm_ij+g.comm_ji+1e-12)
            )),
        })
    return pd.DataFrame(rows)


def freeze_effective_molecular_threshold(
    level0_edge_rel: pd.DataFrame,
    q: float=0.25,
) -> float:
    x=level0_edge_rel.molecular_distance.to_numpy(np.float64)
    x=x[np.isfinite(x)]
    if len(x)==0:
        raise RuntimeError("no finite Level-0 molecular distances")
    return float(np.quantile(x,q))


def evaluate_effective_candidates(
    x: pd.DataFrame,
    thresholds: dict,
    molecular_distance_max: float,
) -> pd.DataFrame:
    if len(x)==0:
        return x.copy()
    y=x.copy()
    y["pass_molecular"]=y.molecular_distance<=molecular_distance_max
    y["pass_mechanics_support"]=(
        y.mechanics_support_fraction
        >= thresholds["min_mechanics_support_fraction"]
    )
    y["pass_tension"]=y.abs_tension_z<=thresholds["abs_tension_z_max"]
    y["pass_delta_p"]=y.abs_delta_p_z<=thresholds["abs_delta_p_z_max"]
    y["pass_communication_strength"]=(
        y.comm_strength>=thresholds["comm_strength_min"]
    )
    y["pass_communication_reciprocity"]=(
        y.comm_reciprocity>=thresholds["comm_reciprocity_min"]
    )
    y["admissible"]=(
        y.pass_molecular
        & y.pass_mechanics_support
        & y.pass_tension
        & y.pass_delta_p
        & y.pass_communication_strength
        & y.pass_communication_reciprocity
    )

    eps=1e-12
    parts=np.vstack([
        (molecular_distance_max-y.molecular_distance)
        /(molecular_distance_max+eps),
        (thresholds["abs_tension_z_max"]-y.abs_tension_z)
        /(thresholds["abs_tension_z_max"]+eps),
        (thresholds["abs_delta_p_z_max"]-y.abs_delta_p_z)
        /(thresholds["abs_delta_p_z_max"]+eps),
        (y.comm_strength-thresholds["comm_strength_min"])
        /(abs(thresholds["comm_strength_min"])+1.0),
        (y.comm_reciprocity-thresholds["comm_reciprocity_min"]),
        y.mechanics_support_fraction-thresholds["min_mechanics_support_fraction"],
    ])
    y["ordering_merit"]=np.nanmean(parts,axis=0)
    return y


def maximal_matching(x: pd.DataFrame) -> pd.DataFrame:
    if len(x)==0:
        z=x.copy(); z["selected"]=False; return z
    a=x[x.admissible].copy()
    if len(a)==0:
        a["selected"]=False; return a
    a=a.sort_values(
        ["ordering_merit","n_boundary_edges","super_i","super_j"],
        ascending=[False,False,True,True],
        kind="mergesort",
    ).reset_index(drop=True)
    used=set();sel=[]
    for r in a.itertuples():
        u=int(r.super_i);v=int(r.super_j)
        take=(u not in used and v not in used)
        sel.append(take)
        if take:
            used.add(u);used.add(v)
    a["selected"]=sel
    return a


def contract(labels: np.ndarray, selected: pd.DataFrame) -> np.ndarray:
    out=labels.copy()
    for r in selected[selected.selected].itertuples():
        a=int(r.super_i);b=int(r.super_j)
        survivor=min(a,b);removed=max(a,b)
        out[out==removed]=survivor
    return out


def failure_profile(x: pd.DataFrame) -> dict:
    if len(x)==0:
        return {
            "fail_molecular":0,
            "fail_mechanics_support":0,
            "fail_tension":0,
            "fail_delta_p":0,
            "fail_communication_strength":0,
            "fail_communication_reciprocity":0,
        }
    return {
        "fail_molecular":int((~x.pass_molecular).sum()),
        "fail_mechanics_support":int((~x.pass_mechanics_support).sum()),
        "fail_tension":int((~x.pass_tension).sum()),
        "fail_delta_p":int((~x.pass_delta_p).sum()),
        "fail_communication_strength":int(
            (~x.pass_communication_strength).sum()
        ),
        "fail_communication_reciprocity":int(
            (~x.pass_communication_reciprocity).sum()
        ),
    }


def node_statistics(
    labels: np.ndarray,
    xy: np.ndarray,
    states: np.ndarray,
    node_ids: np.ndarray,
    centroids: np.ndarray,
    radii: np.ndarray,
) -> pd.DataFrame:
    sizes=pd.Series(labels).value_counts().sort_index()
    rows=[]
    idx={int(x):k for k,x in enumerate(node_ids)}
    for node,n in sizes.items():
        k=idx[int(node)]
        q=np.flatnonzero(labels==node)
        within=0.0
        if len(q)>1:
            # RMS molecular spread around effective state.
            # state matrix uses current node ordering.
            # caller can attach a separate dispersion if desired.
            within=np.nan
        rows.append({
            "hierarchy_node":int(node),
            "n_level0_cells":int(n),
            "centroid_x":float(centroids[k,0]),
            "centroid_y":float(centroids[k,1]),
            "radius":float(radii[k]),
        })
    return pd.DataFrame(rows)


def summarize_level(
    level:int,
    labels_before:np.ndarray,
    labels_after:np.ndarray,
    cand:pd.DataFrame,
    selected:pd.DataFrame,
    node_stats:pd.DataFrame,
) -> dict:
    n_before=len(np.unique(labels_before))
    n_after=len(np.unique(labels_after))
    adm=int(cand.admissible.sum()) if len(cand) else 0
    sel=int(selected.selected.sum()) if len(selected) else 0

    sizes=node_stats.n_level0_cells.to_numpy(np.float64)
    radii=node_stats.radius.to_numpy(np.float64)

    out={
        "level":int(level),
        "n_nodes_before":int(n_before),
        "n_nodes_after":int(n_after),
        "n_superedges":int(len(cand)),
        "n_admissible":adm,
        "n_selected":sel,
        "admissible_fraction":float(adm/len(cand)) if len(cand) else 0.0,
        "selected_fraction_of_admissible":float(sel/adm) if adm else 0.0,
        "node_size_mean":float(np.mean(sizes)) if len(sizes) else np.nan,
        "node_size_median":float(np.median(sizes)) if len(sizes) else np.nan,
        "node_size_q95":float(np.quantile(sizes,.95)) if len(sizes) else np.nan,
        "node_size_max":float(np.max(sizes)) if len(sizes) else np.nan,
        "radius_mean":float(np.mean(radii)) if len(radii) else np.nan,
        "radius_median":float(np.median(radii)) if len(radii) else np.nan,
        "radius_q95":float(np.quantile(radii,.95)) if len(radii) else np.nan,
        "radius_max":float(np.max(radii)) if len(radii) else np.nan,
        **failure_profile(cand),
    }
    if len(cand):
        out.update({
            "molecular_distance_median":float(np.nanmedian(cand.molecular_distance)),
            "mechanics_support_median":float(np.nanmedian(cand.mechanics_support_fraction)),
            "abs_tension_z_median":float(np.nanmedian(cand.abs_tension_z)),
            "abs_delta_p_z_median":float(np.nanmedian(cand.abs_delta_p_z)),
            "comm_strength_median":float(np.nanmedian(cand.comm_strength)),
            "comm_reciprocity_median":float(np.nanmedian(cand.comm_reciprocity)),
            "comm_asymmetry_median":float(np.nanmedian(cand.comm_asymmetry)),
        })
    return out


__all__=[
    "FlowConfig",
    "aggregate_supernode_expression",
    "aggregate_supernode_centroids",
    "aggregate_supernode_radii",
    "current_boundary_relations",
    "freeze_effective_molecular_threshold",
    "evaluate_effective_candidates",
    "maximal_matching",
    "contract",
    "node_statistics",
    "summarize_level",
]
