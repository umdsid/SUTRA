"""Conservative short-hierarchy pilot for STRATA v0.7.2.

Scientific rules
----------------
* Only measured Level-0 cell-cell contacts can support contraction.
* Functional molecular, primitive mechanics, and directed communication gates
  are conjunctive. No block can compensate for failure of another block.
* Missing mechanics is missing, never zero.
* All thresholds are derived once from Level-0 and then frozen.
* Later levels aggregate frozen Level-0 edge relations; they never renormalize.
* The pilot performs only a few aggressively capped levels.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import defaultdict
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PilotConfig:
    n_levels: int = 3
    max_pair_fraction: float = 0.005
    max_pairs_per_level: int = 250
    min_mechanics_support_fraction: float = 0.50

    # Level-0 quantile rules. These are frozen before level 1.
    molecular_max_quantile: float = 0.25
    mechanics_abs_max_quantile: float = 0.75
    communication_strength_min_quantile: float = 0.50
    communication_reciprocity_min_quantile: float = 0.50

    eps: float = 1e-12


def _q(x, q):
    x=np.asarray(x,dtype=np.float64)
    x=x[np.isfinite(x)]
    if len(x)==0:
        return np.nan
    return float(np.quantile(x,q))


def communication_relations(
    Y: np.ndarray,
    edges: pd.DataFrame,
    genes: tuple[str,...],
    registry: pd.DataFrame,
    eps: float=1e-12,
) -> pd.DataFrame:
    """
    Directed communication on measured contact candidates.

    Contact and diffusible interaction families remain separate while scored.
    Because hierarchy contractions themselves are restricted to measured
    contacts, no extra spatial kernel is invented for the pilot.

    For one interaction r=(sender,receiver), support is
        sqrt(Y_i,sender * Y_j,receiver)
    on the already normalized nonnegative Level-0 molecular representation.
    Within each spatial kind, interactions are averaged equally. The two kinds
    are then averaged equally when both exist.

    Reciprocity is:
        1 - |s_ij-s_ji|/(s_ij+s_ji+eps)
    so 1 is reciprocal and 0 is strongly one-way.
    """
    idx={str(g):k for k,g in enumerate(genes)}
    reg=registry.copy()
    if "panel_supported" in reg:
        reg=reg[reg.panel_supported.astype(bool)]
    reg=reg[
        reg.sender_gene.astype(str).isin(idx)
        & reg.receiver_gene.astype(str).isin(idx)
        & reg.kind.astype(str).isin(["contact","diffusible"])
    ].copy()

    i=edges.cell_i_index.to_numpy(np.int64)
    j=edges.cell_j_index.to_numpy(np.int64)

    out={
        "comm_contact_ij":np.full(len(edges),np.nan),
        "comm_contact_ji":np.full(len(edges),np.nan),
        "comm_diffusible_ij":np.full(len(edges),np.nan),
        "comm_diffusible_ji":np.full(len(edges),np.nan),
    }

    kinds_present=[]
    for kind in ("contact","diffusible"):
        sub=reg[reg.kind.astype(str)==kind]
        if len(sub)==0:
            continue
        kinds_present.append(kind)
        fij=np.zeros(len(edges),dtype=np.float64)
        fji=np.zeros(len(edges),dtype=np.float64)
        for r in sub.itertuples():
            s=idx[str(r.sender_gene)]
            t=idx[str(r.receiver_gene)]
            fij += np.sqrt(
                np.maximum(0.0,Y[i,s].astype(np.float64))
                * np.maximum(0.0,Y[j,t].astype(np.float64))
            )
            fji += np.sqrt(
                np.maximum(0.0,Y[j,s].astype(np.float64))
                * np.maximum(0.0,Y[i,t].astype(np.float64))
            )
        fij/=len(sub); fji/=len(sub)
        out[f"comm_{kind}_ij"]=fij
        out[f"comm_{kind}_ji"]=fji

    if not kinds_present:
        raise RuntimeError("no panel-supported typed communication interactions")

    ij_parts=[out[f"comm_{k}_ij"] for k in kinds_present]
    ji_parts=[out[f"comm_{k}_ji"] for k in kinds_present]
    sij=np.nanmean(np.vstack(ij_parts),axis=0)
    sji=np.nanmean(np.vstack(ji_parts),axis=0)

    strength=np.log1p(sij+sji)
    reciprocity=1.0-np.abs(sij-sji)/(sij+sji+eps)
    reciprocity=np.clip(reciprocity,0.0,1.0)

    R=pd.DataFrame(out)
    R["comm_ij"]=sij
    R["comm_ji"]=sji
    R["comm_strength"]=strength
    R["comm_reciprocity"]=reciprocity
    R["n_panel_interactions"]=int(len(reg))
    R["n_spatial_kinds"]=int(len(kinds_present))
    return R


def freeze_thresholds(edge_rel: pd.DataFrame, cfg: PilotConfig) -> dict:
    """
    Freeze all admissibility thresholds from Level-0 only.
    """
    complete=(
        edge_rel.mechanics_tension_valid.astype(bool)
        & edge_rel.mechanics_delta_p_valid.astype(bool)
    )
    tau_abs=np.abs(edge_rel.loc[complete,"tension_z"].to_numpy(np.float64))
    dp_abs=np.abs(edge_rel.loc[complete,"delta_pressure_z"].to_numpy(np.float64))

    thresholds={
        "molecular_z_max":_q(
            edge_rel.molecular_z,cfg.molecular_max_quantile
        ),
        "abs_tension_z_max":_q(
            tau_abs,cfg.mechanics_abs_max_quantile
        ),
        "abs_delta_p_z_max":_q(
            dp_abs,cfg.mechanics_abs_max_quantile
        ),
        "comm_strength_min":_q(
            edge_rel.comm_strength,cfg.communication_strength_min_quantile
        ),
        "comm_reciprocity_min":_q(
            edge_rel.comm_reciprocity,
            cfg.communication_reciprocity_min_quantile,
        ),
        "min_mechanics_support_fraction":cfg.min_mechanics_support_fraction,
        "source":"Level-0 only",
        "recomputed_at_later_levels":False,
        "quantile_rules":{
            "molecular_max":cfg.molecular_max_quantile,
            "mechanics_abs_max":cfg.mechanics_abs_max_quantile,
            "communication_strength_min":cfg.communication_strength_min_quantile,
            "communication_reciprocity_min":cfg.communication_reciprocity_min_quantile,
        },
    }
    if not all(np.isfinite(v) for k,v in thresholds.items() if k.endswith(("_max","_min"))):
        raise RuntimeError("nonfinite Level-0 hierarchy threshold")
    return thresholds


def summarize_superedges(
    edge_rel: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Aggregate frozen Level-0 edge relations to current supernode boundaries.

    Medians are used for block summaries. Mechanics support is reported
    separately, so absent mechanics cannot be converted to a numerical zero.
    """
    u=labels[edge_rel.cell_i_index.to_numpy(np.int64)]
    v=labels[edge_rel.cell_j_index.to_numpy(np.int64)]
    lo=np.minimum(u,v); hi=np.maximum(u,v)
    keep=lo!=hi
    if not np.any(keep):
        return pd.DataFrame()

    W=edge_rel.loc[keep].copy()
    W["super_i"]=lo[keep]
    W["super_j"]=hi[keep]
    W["tau_abs"]=np.abs(W.tension_z)
    W["dp_abs"]=np.abs(W.delta_pressure_z)
    W["mech_complete"]=(
        W.mechanics_tension_valid.astype(bool)
        & W.mechanics_delta_p_valid.astype(bool)
    )

    rows=[]
    for (a,b),g in W.groupby(["super_i","super_j"],sort=False):
        mech=g[g.mech_complete]
        rows.append({
            "super_i":int(a),
            "super_j":int(b),
            "n_boundary_edges":int(len(g)),
            "molecular_z":float(np.nanmedian(g.molecular_z)),
            "comm_strength":float(np.nanmedian(g.comm_strength)),
            "comm_reciprocity":float(np.nanmedian(g.comm_reciprocity)),
            "mechanics_support_fraction":float(g.mech_complete.mean()),
            "abs_tension_z":(
                float(np.nanmedian(mech.tau_abs)) if len(mech) else np.nan
            ),
            "abs_delta_p_z":(
                float(np.nanmedian(mech.dp_abs)) if len(mech) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def evaluate_candidates(superedges: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    if len(superedges)==0:
        return superedges.copy()

    x=superedges.copy()
    x["pass_molecular"]=x.molecular_z <= thresholds["molecular_z_max"]
    x["pass_mechanics_support"]=(
        x.mechanics_support_fraction
        >= thresholds["min_mechanics_support_fraction"]
    )
    x["pass_tension"]=x.abs_tension_z <= thresholds["abs_tension_z_max"]
    x["pass_delta_p"]=x.abs_delta_p_z <= thresholds["abs_delta_p_z_max"]
    x["pass_communication_strength"]=(
        x.comm_strength >= thresholds["comm_strength_min"]
    )
    x["pass_communication_reciprocity"]=(
        x.comm_reciprocity >= thresholds["comm_reciprocity_min"]
    )
    x["admissible"]=(
        x.pass_molecular
        & x.pass_mechanics_support
        & x.pass_tension
        & x.pass_delta_p
        & x.pass_communication_strength
        & x.pass_communication_reciprocity
    )

    # Ordering merit is used only after conjunctive admissibility.
    # Each term is a positive margin relative to its already-frozen gate.
    eps=1e-12
    mol=(
        thresholds["molecular_z_max"]-x.molecular_z
    )/(abs(thresholds["molecular_z_max"])+1.0)
    tau=(
        thresholds["abs_tension_z_max"]-x.abs_tension_z
    )/(thresholds["abs_tension_z_max"]+eps)
    dp=(
        thresholds["abs_delta_p_z_max"]-x.abs_delta_p_z
    )/(thresholds["abs_delta_p_z_max"]+eps)
    cs=(
        x.comm_strength-thresholds["comm_strength_min"]
    )/(abs(thresholds["comm_strength_min"])+1.0)
    cr=(
        x.comm_reciprocity-thresholds["comm_reciprocity_min"]
    )/(1.0+eps)
    ms=x.mechanics_support_fraction-thresholds["min_mechanics_support_fraction"]

    x["ordering_merit"]=np.nanmean(
        np.vstack([mol,tau,dp,cs,cr,ms]),axis=0
    )
    return x


def greedy_matching(
    candidates: pd.DataFrame,
    n_active_nodes: int,
    cfg: PilotConfig,
) -> pd.DataFrame:
    """
    Select a deterministic disjoint set of admissible contractions.
    """
    A=candidates[candidates.admissible].copy()
    if len(A)==0:
        A["selected"]=False
        return A

    cap=min(
        cfg.max_pairs_per_level,
        max(1,int(math.ceil(cfg.max_pair_fraction*n_active_nodes))),
    )

    A=A.sort_values(
        ["ordering_merit","n_boundary_edges","super_i","super_j"],
        ascending=[False,False,True,True],
        kind="mergesort",
    ).reset_index(drop=True)

    used=set()
    selected=[]
    n_selected=0
    for r in A.itertuples():
        u=int(r.super_i);v=int(r.super_j)
        take=(
            u not in used
            and v not in used
            and n_selected < cap
        )
        selected.append(bool(take))
        if take:
            used.add(u)
            used.add(v)
            n_selected += 1
    A["selected"]=selected
    return A


def contract_labels(labels: np.ndarray, selected: pd.DataFrame) -> tuple[np.ndarray,dict]:
    """
    Deterministic contraction: smaller current label survives.
    """
    out=labels.copy()
    mapping={}
    if len(selected):
        for r in selected[selected.selected].itertuples():
            a=int(r.super_i);b=int(r.super_j)
            survivor=min(a,b);removed=max(a,b)
            mapping[removed]=survivor

    # Selected pairs are disjoint, so one replacement pass is sufficient.
    for removed,survivor in mapping.items():
        out[out==removed]=survivor

    return out,mapping


def membership_table(labels: np.ndarray, level: int) -> pd.DataFrame:
    return pd.DataFrame({
        "cell_index":np.arange(len(labels),dtype=np.int64),
        "hierarchy_node":labels.astype(np.int64),
        "level":int(level),
    })


def level_summary(level,candidates,selected,labels_before,labels_after):
    return {
        "level":int(level),
        "n_nodes_before":int(len(np.unique(labels_before))),
        "n_superedges":int(len(candidates)),
        "n_admissible":int(candidates.admissible.sum()) if len(candidates) else 0,
        "n_selected":int(selected.selected.sum()) if len(selected) else 0,
        "n_nodes_after":int(len(np.unique(labels_after))),
    }


__all__=[
    "PilotConfig",
    "communication_relations",
    "freeze_thresholds",
    "summarize_superedges",
    "evaluate_candidates",
    "greedy_matching",
    "contract_labels",
    "membership_table",
    "level_summary",
]
