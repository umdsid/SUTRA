"""Full STRATA hierarchy engine.

The scientific admissibility rules are inherited unchanged from v0.7.2.
The only pilot restriction removed is the artificial per-level contraction cap.

Production semantics
--------------------
* Thresholds are frozen from Level-0.
* Candidate support remains measured Level-0 cell-cell contact only.
* Molecular, primitive-mechanics and directed-communication gates remain
  conjunctive.
* Missing mechanics remains missing.
* Each level takes a deterministic maximal disjoint matching over the currently
  admissible supernode boundaries.
* The trajectory stops naturally when no admissible disjoint contraction
  remains.
* A hard level ceiling is safety-only. Reaching it with available contractions
  is HOLD, never PASS.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from sutra.hierarchy.v072.pilot import evaluate_candidates


@dataclass(frozen=True)
class FullConfig:
    max_levels: int = 250
    checkpoint_every: int = 5
    min_boundary_edges: int = 1


def maximal_disjoint_matching(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic maximal matching among conjunctively admissible boundaries.

    Ordering merit does not alter admissibility; it only chooses among
    simultaneously admissible competing contractions.
    """
    if len(candidates)==0:
        z=candidates.copy()
        z["selected"]=False
        return z

    A=candidates[candidates.admissible].copy()
    if len(A)==0:
        A["selected"]=False
        return A

    A=A.sort_values(
        ["ordering_merit","n_boundary_edges","super_i","super_j"],
        ascending=[False,False,True,True],
        kind="mergesort",
    ).reset_index(drop=True)

    used=set()
    chosen=[]
    for r in A.itertuples():
        u=int(r.super_i); v=int(r.super_j)
        take=(u not in used and v not in used)
        chosen.append(take)
        if take:
            used.add(u); used.add(v)
    A["selected"]=chosen
    return A


def contract_labels_inplace(labels: np.ndarray, selected: pd.DataFrame):
    """
    Contract selected disjoint pairs; lower node ID survives.
    """
    if len(selected)==0:
        return labels, pd.DataFrame(
            columns=["removed_node","survivor_node"]
        )

    rows=[]
    for r in selected[selected.selected].itertuples():
        a=int(r.super_i); b=int(r.super_j)
        survivor=min(a,b); removed=max(a,b)
        labels[labels==removed]=survivor
        rows.append({
            "removed_node":removed,
            "survivor_node":survivor,
        })
    return labels,pd.DataFrame(rows)


def failure_counts(candidates: pd.DataFrame) -> dict:
    """
    Count gate failures independently; counts can overlap by design.
    """
    if len(candidates)==0:
        return {
            "fail_molecular":0,
            "fail_mechanics_support":0,
            "fail_tension":0,
            "fail_delta_p":0,
            "fail_communication_strength":0,
            "fail_communication_reciprocity":0,
        }
    return {
        "fail_molecular":int((~candidates.pass_molecular).sum()),
        "fail_mechanics_support":int(
            (~candidates.pass_mechanics_support).sum()
        ),
        "fail_tension":int((~candidates.pass_tension).sum()),
        "fail_delta_p":int((~candidates.pass_delta_p).sum()),
        "fail_communication_strength":int(
            (~candidates.pass_communication_strength).sum()
        ),
        "fail_communication_reciprocity":int(
            (~candidates.pass_communication_reciprocity).sum()
        ),
    }


def hierarchy_node_sizes(labels: np.ndarray) -> pd.DataFrame:
    ids,n=np.unique(labels,return_counts=True)
    return pd.DataFrame({
        "hierarchy_node":ids.astype(np.int64),
        "n_level0_cells":n.astype(np.int64),
    })


def size_summary(labels: np.ndarray) -> dict:
    _,n=np.unique(labels,return_counts=True)
    if len(n)==0:
        return {}
    return {
        "n_nodes":int(len(n)),
        "size_min":int(n.min()),
        "size_median":float(np.median(n)),
        "size_mean":float(np.mean(n)),
        "size_q95":float(np.quantile(n,.95)),
        "size_max":int(n.max()),
        "n_singletons":int((n==1).sum()),
        "fraction_cells_in_non_singletons":float(
            n[n>1].sum()/n.sum()
        ),
    }


def natural_stop_reason(
    n_superedges:int,
    n_admissible:int,
    n_selected:int,
) -> str | None:
    if n_superedges==0:
        return "no_remaining_supernode_boundaries"
    if n_admissible==0:
        return "no_admissible_boundaries"
    if n_selected==0:
        return "no_disjoint_admissible_contractions"
    return None


__all__=[
    "FullConfig",
    "maximal_disjoint_matching",
    "contract_labels_inplace",
    "failure_counts",
    "hierarchy_node_sizes",
    "size_summary",
    "natural_stop_reason",
]
