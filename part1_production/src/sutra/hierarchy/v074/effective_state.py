"""STRATA v0.7.4.1 effective-state hierarchy utilities.

Communication semantics:
* zero/near-zero support is UNSUPPORTED, not reciprocal;
* reciprocity is defined only above a frozen positive support floor;
* support floor is estimated from positive Level-0 communication support only;
* unsupported boundaries fail the communication gate without being assigned
  artificial zero or one-valued reciprocity.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FlowConfig:
    max_levels: int = 250
    checkpoint_every: int = 2
    min_mechanics_support_fraction: float = 0.50
    communication_support_positive_quantile: float = 0.10
    eps: float = 1e-12


def freeze_communication_support_floor(
    edge_rel: pd.DataFrame,
    positive_quantile: float = 0.10,
    eps: float = 1e-12,
) -> dict:
    """
    Freeze a positive Level-0 communication support floor.

    Uses raw bidirectional support:
        support = comm_ij + comm_ji

    Only strictly positive finite support values contribute to the quantile.
    Zero-support edges remain a distinct unsupported class.
    """
    support = (
        edge_rel.comm_ij.to_numpy(np.float64)
        + edge_rel.comm_ji.to_numpy(np.float64)
    )
    finite = support[np.isfinite(support)]
    positive = finite[finite > eps]

    if len(positive) == 0:
        raise RuntimeError(
            "no positive Level-0 communication support exists; "
            "directed communication block cannot be certified"
        )

    floor = float(np.quantile(positive, positive_quantile))
    floor = max(floor, eps)

    return {
        "communication_support_floor": floor,
        "positive_quantile": float(positive_quantile),
        "n_level0_edges": int(len(support)),
        "n_finite_support": int(len(finite)),
        "n_positive_support": int(len(positive)),
        "n_zero_or_nearzero_support": int(np.sum(finite <= eps)),
        "positive_support_fraction": float(len(positive) / len(finite))
            if len(finite) else 0.0,
        "source": "positive Level-0 bidirectional communication support",
        "recomputed_at_later_levels": False,
    }


def communication_support_and_reciprocity(
    comm_ij: np.ndarray,
    comm_ji: np.ndarray,
    support_floor: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Return support, supported-mask, reciprocity.

    Reciprocity is NA below support floor.
    """
    sij=np.asarray(comm_ij,dtype=np.float64)
    sji=np.asarray(comm_ji,dtype=np.float64)
    support=sij+sji
    supported=np.isfinite(support) & (support >= support_floor)

    recip=np.full(len(support),np.nan,dtype=np.float64)
    q=supported & np.isfinite(sij) & np.isfinite(sji)
    recip[q]=1.0-np.abs(sij[q]-sji[q])/(support[q]+eps)
    recip[q]=np.clip(recip[q],0.0,1.0)
    return support,supported,recip


def aggregate_supernode_expression(
    Y: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
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



def current_boundary_relations_v074(
    edge_rel: pd.DataFrame,
    labels: np.ndarray,
    node_ids: np.ndarray,
    states: np.ndarray,
    L: np.ndarray,
    lam: float,
) -> pd.DataFrame:
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

def current_boundary_relations(
    edge_rel: pd.DataFrame,
    labels: np.ndarray,
    node_ids: np.ndarray,
    states: np.ndarray,
    L: np.ndarray,
    lam: float,
    communication_support_floor: float,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Recompute effective molecular separation and current boundary summaries.

    Communication uses the median directional signals over supporting Level-0
    interfaces, then applies the explicit support floor before reciprocity is
    defined.
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

        sij=float(np.nanmedian(g.comm_ij))
        sji=float(np.nanmedian(g.comm_ji))
        support=np.asarray([sij+sji],dtype=np.float64)
        _,supported,recip=communication_support_and_reciprocity(
            np.asarray([sij]),np.asarray([sji]),
            communication_support_floor,eps
        )

        comm_strength=float(np.log1p(max(0.0,support[0])))
        comm_asym=np.nan
        if supported[0]:
            comm_asym=float(abs(sij-sji)/(support[0]+eps))

        rows.append({
            "super_i":int(a),
            "super_j":int(b),
            "n_boundary_edges":int(len(g)),
            "molecular_distance":float(dm),
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
            "comm_ij":sij,
            "comm_ji":sji,
            "comm_support":float(support[0]),
            "comm_supported":bool(supported[0]),
            "comm_strength":comm_strength,
            "comm_reciprocity":float(recip[0]) if np.isfinite(recip[0]) else np.nan,
            "comm_asymmetry":comm_asym,
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


def freeze_supported_reciprocity_threshold(
    edge_rel: pd.DataFrame,
    support_floor: float,
    q: float=0.50,
    eps: float=1e-12,
) -> dict:
    _,supported,recip=communication_support_and_reciprocity(
        edge_rel.comm_ij.to_numpy(np.float64),
        edge_rel.comm_ji.to_numpy(np.float64),
        support_floor,eps
    )
    x=recip[supported & np.isfinite(recip)]
    if len(x)==0:
        raise RuntimeError(
            "no supported Level-0 communication edges have finite reciprocity"
        )
    return {
        "comm_reciprocity_min":float(np.quantile(x,q)),
        "source_quantile":float(q),
        "n_supported_level0_edges":int(len(x)),
        "source":"supported Level-0 communication edges only",
        "recomputed_at_later_levels":False,
    }


def evaluate_effective_candidates(
    x: pd.DataFrame,
    thresholds: dict,
    molecular_distance_max: float,
    comm_reciprocity_min: float,
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

    # Communication now has an explicit observability/support gate.
    y["pass_communication_support"]=y.comm_supported.astype(bool)
    y["pass_communication_reciprocity"]=(
        y.comm_supported.astype(bool)
        & np.isfinite(y.comm_reciprocity)
        & (y.comm_reciprocity>=comm_reciprocity_min)
    )

    y["admissible"]=(
        y.pass_molecular
        & y.pass_mechanics_support
        & y.pass_tension
        & y.pass_delta_p
        & y.pass_communication_support
        & y.pass_communication_reciprocity
    )

    eps=1e-12
    recip_margin=np.where(
        y.comm_supported,
        y.comm_reciprocity-comm_reciprocity_min,
        -1.0,
    )
    support_margin=np.where(
        y.comm_supported,
        np.log1p(y.comm_support),
        -1.0,
    )
    parts=np.vstack([
        (molecular_distance_max-y.molecular_distance)
        /(molecular_distance_max+eps),
        (thresholds["abs_tension_z_max"]-y.abs_tension_z)
        /(thresholds["abs_tension_z_max"]+eps),
        (thresholds["abs_delta_p_z_max"]-y.abs_delta_p_z)
        /(thresholds["abs_delta_p_z_max"]+eps),
        support_margin,
        recip_margin,
        y.mechanics_support_fraction-thresholds["min_mechanics_support_fraction"],
    ])
    y["ordering_merit"]=np.nanmean(parts,axis=0)
    return y



def evaluate_effective_candidates_v074(
    x: pd.DataFrame,
    thresholds: dict,
    molecular_distance_max: float,
) -> pd.DataFrame:
    """Frozen v0.7.4 effective candidate gate using exact pre-v0.7.4.1 semantics."""
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
            "fail_communication_support":0,
            "fail_communication_reciprocity":0,
        }
    return {
        "fail_molecular":int((~x.pass_molecular).sum()),
        "fail_mechanics_support":int((~x.pass_mechanics_support).sum()),
        "fail_tension":int((~x.pass_tension).sum()),
        "fail_delta_p":int((~x.pass_delta_p).sum()),
        "fail_communication_support":int((~x.pass_communication_support).sum()),
        "fail_communication_reciprocity":int(
            (~x.pass_communication_reciprocity).sum()
        ),
    }


def node_statistics(
    labels: np.ndarray,
    xy: np.ndarray,
    node_ids: np.ndarray,
    centroids: np.ndarray,
    radii: np.ndarray,
) -> pd.DataFrame:
    sizes=pd.Series(labels).value_counts().sort_index()
    rows=[]
    idx={int(x):k for k,x in enumerate(node_ids)}
    for node,n in sizes.items():
        k=idx[int(node)]
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
        supported=cand[cand.comm_supported.astype(bool)]
        out.update({
            "molecular_distance_median":float(np.nanmedian(cand.molecular_distance)),
            "mechanics_support_median":float(np.nanmedian(cand.mechanics_support_fraction)),
            "abs_tension_z_median":float(np.nanmedian(cand.abs_tension_z)),
            "abs_delta_p_z_median":float(np.nanmedian(cand.abs_delta_p_z)),
            "comm_support_median":float(np.nanmedian(cand.comm_support)),
            "comm_supported_fraction":float(cand.comm_supported.mean()),
            "comm_reciprocity_supported_median":(
                float(np.nanmedian(supported.comm_reciprocity))
                if len(supported) else np.nan
            ),
            "comm_asymmetry_supported_median":(
                float(np.nanmedian(supported.comm_asymmetry))
                if len(supported) else np.nan
            ),
        })
    return out


__all__=[
    "FlowConfig",
    "freeze_communication_support_floor",
    "communication_support_and_reciprocity",
    "aggregate_supernode_expression",
    "aggregate_supernode_centroids",
    "aggregate_supernode_radii",
    "functional_distance_between_states",
    "current_boundary_relations",
    "freeze_effective_molecular_threshold",
    "freeze_supported_reciprocity_threshold",
    "evaluate_effective_candidates",
    "maximal_matching",
    "contract",
    "node_statistics",
    "summarize_level",
]
