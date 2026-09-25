"""STRATA v0.9.0 ultraslow effective-state flow.

Scientific gates are inherited from the frozen Level-0 construction.
Only the contraction schedule changes.

Principles
----------
1. Recompute the current effective state after every microstep.
2. Use tiny contraction batches by default.
3. Search multiple deterministic disjoint matchings at every step.
4. Near exhaustion, enumerate all disjoint matchings exactly on the small
   eligible frontier before declaring that no contraction route remains.
5. Never relax molecular, mechanics, communication, or geometry-resolution
   requirements.
6. Larger batches are a runtime rescue only; they are not a scientific fallback.
7. Natural exhaustion means zero eligible boundaries after full recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass,asdict
from itertools import combinations
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SlowFlowConfig:
    epsilon_fraction: float = 2.0e-4
    soft_step_1: int = 1500
    soft_step_2: int = 3000
    soft_step_3: int = 5000
    rescue_fraction_1: float = 5.0e-4
    rescue_fraction_2: float = 1.0e-3
    last_resort_fraction: float = 5.0e-3
    exact_frontier_max_edges: int = 18
    checkpoint_every: int = 50
    heavy_landmark_reduction_fraction: float = 2.5e-3
    max_microsteps_safety: int = 20000
    archive_all_candidates: bool = True

    def validate(self):
        vals=[
            self.epsilon_fraction,self.rescue_fraction_1,
            self.rescue_fraction_2,self.last_resort_fraction
        ]
        if any((x<=0 or x>=.1) for x in vals):
            raise ValueError("flow fractions must lie in (0,0.1)")
        if not (
            self.epsilon_fraction
            <= self.rescue_fraction_1
            <= self.rescue_fraction_2
            <= self.last_resort_fraction
        ):
            raise ValueError("flow fractions must be nondecreasing")
        if self.exact_frontier_max_edges<1:
            raise ValueError("exact frontier must be positive")
        return self


def batch_fraction(step,cfg):
    """Adaptive step size: almost-continuous first, larger only as rescue."""
    if step < cfg.soft_step_1:
        return cfg.epsilon_fraction,"epsilon"
    if step < cfg.soft_step_2:
        return cfg.rescue_fraction_1,"rescue_1"
    if step < cfg.soft_step_3:
        return cfg.rescue_fraction_2,"rescue_2"
    return cfg.last_resort_fraction,"last_resort"


def batch_cap(n_nodes,step,cfg):
    frac,mode=batch_fraction(step,cfg)
    # At least one pair; never remove more than the requested node fraction.
    cap=max(1,int(math.ceil(float(n_nodes)*frac)))
    cap=min(cap,max(1,n_nodes//2))
    return cap,frac,mode


def eligible_boundaries(cand):
    if len(cand)==0:
        return cand.copy()
    q=(
        cand.admissible.astype(bool)
        & cand.geometry_state_resolved.astype(bool)
        & np.isfinite(cand.geometry_pair_cost)
        & np.isfinite(cand.alpha_cost)
    )
    return cand[q].copy()


def _pair_tuple(r):
    u=int(r.super_i);v=int(r.super_j)
    return (min(u,v),max(u,v))


def _greedy_order(df,cols,ascending,cap):
    x=df.sort_values(cols,ascending=ascending,kind="mergesort")
    used=set();rows=[]
    for _,r in x.iterrows():
        u,v=_pair_tuple(r)
        if u in used or v in used:
            continue
        rows.append(r)
        used.add(u);used.add(v)
        if len(rows)>=cap:
            break
    return pd.DataFrame(rows,columns=df.columns)


def candidate_matchings(df,cap):
    """Multiple deterministic routes through the same frozen eligible set."""
    if len(df)==0:
        return []

    specs=[
        (
            "geometry_first",
            ["geometry_pair_cost","alpha_cost","super_i","super_j"],
            [True,True,True,True],
        ),
        (
            "symmetric_first",
            ["alpha_cost","geometry_pair_cost","super_i","super_j"],
            [True,True,True,True],
        ),
        (
            "directional_conservative",
            ["directional_correction_abs","geometry_pair_cost","super_i","super_j"],
            [True,True,True,True],
        ),
        (
            "biology_merit",
            ["ordering_merit","geometry_pair_cost","super_i","super_j"],
            [False,True,True,True],
        ),
        (
            "boundary_supported",
            ["n_boundary_edges","geometry_pair_cost","super_i","super_j"],
            [False,True,True,True],
        ),
    ]

    x=df.copy()
    if "directional_correction_abs" not in x.columns:
        if "directional_correction" in x.columns:
            x["directional_correction_abs"]=x.directional_correction.abs()
        else:
            x["directional_correction_abs"]=0.0

    out=[]
    for name,cols,asc in specs:
        cols=[c for c in cols if c in x.columns]
        asc=asc[:len(cols)]
        y=_greedy_order(x,cols,asc,cap)
        out.append((name,y))
    return out


def matching_score(name,batch,cap):
    """Lexicographic damage score; lower is preferred.

    First maximize realized batch size, then minimize total/max geometric cost,
    then preserve high biological merit, then deterministic pair IDs.
    """
    n=len(batch)
    if n==0:
        return (1,cap,float("inf"),float("inf"),float("inf"),name)

    g=batch.geometry_pair_cost.to_numpy(float)
    merit=(
        batch.ordering_merit.to_numpy(float)
        if "ordering_merit" in batch.columns
        else np.zeros(n)
    )
    ids=tuple(sorted(_pair_tuple(r) for _,r in batch.iterrows()))
    return (
        0 if n>=cap else 1,
        cap-n,
        float(np.sum(g)),
        float(np.max(g)),
        -float(np.sum(merit)),
        ids,
        name,
    )


def exact_best_matching(df,cap):
    """Exact search over all disjoint matchings on a small frontier.

    This is intentionally used near exhaustion. Objective:
      1. maximize cardinality up to cap;
      2. minimize total geometric cost;
      3. minimize maximum geometric cost;
      4. deterministic pair IDs.
    """
    if len(df)==0:
        return pd.DataFrame(columns=df.columns)

    records=[r for _,r in df.iterrows()]
    best_rows=[]
    best_score=None

    def rec(k,used,chosen):
        nonlocal best_rows,best_score

        if len(chosen)>=cap or k>=len(records):
            if not chosen:
                score=(0,float("inf"),float("inf"),())
            else:
                g=np.array([float(r.geometry_pair_cost) for r in chosen])
                ids=tuple(sorted(_pair_tuple(r) for r in chosen))
                score=(-len(chosen),float(g.sum()),float(g.max()),ids)
            if best_score is None or score<best_score:
                best_score=score
                best_rows=list(chosen)
            return

        # optimistic cardinality pruning
        remaining=len(records)-k
        max_possible=len(chosen)+remaining
        if best_score is not None and -min(cap,max_possible)>best_score[0]:
            pass

        r=records[k]
        u,v=_pair_tuple(r)

        # include
        if u not in used and v not in used:
            rec(k+1,used|{u,v},chosen+[r])

        # exclude
        rec(k+1,used,chosen)

    rec(0,set(),[])
    return pd.DataFrame(best_rows,columns=df.columns)


def choose_slow_batch(cand,n_nodes,step,cfg):
    elig=eligible_boundaries(cand)
    cap,frac,mode=batch_cap(n_nodes,step,cfg)

    if len(elig)==0:
        return {
            "eligible":elig,
            "selected":pd.DataFrame(columns=cand.columns),
            "selection_mode":"none",
            "batch_cap":cap,
            "batch_fraction":frac,
            "step_mode":mode,
            "routes_considered":0,
            "exact_frontier":False,
        }

    if len(elig)<=cfg.exact_frontier_max_edges:
        y=exact_best_matching(elig,cap)
        return {
            "eligible":elig,
            "selected":y,
            "selection_mode":"exact_frontier",
            "batch_cap":cap,
            "batch_fraction":frac,
            "step_mode":mode,
            "routes_considered":"all_disjoint_matchings",
            "exact_frontier":True,
        }

    routes=candidate_matchings(elig,cap)
    scored=[
        (matching_score(name,batch,cap),name,batch)
        for name,batch in routes
    ]
    scored.sort(key=lambda z:z[0])
    _,name,y=scored[0]

    # Mathematical last safeguard: if an eligible edge exists, a one-edge
    # matching always exists. Never declare exhaustion because a heuristic route
    # happened to fail.
    if len(y)==0:
        z=elig.sort_values(
            ["geometry_pair_cost","super_i","super_j"],
            ascending=[True,True,True],
            kind="mergesort",
        ).head(1)
        name="single_edge_guarantee"
        y=z

    return {
        "eligible":elig,
        "selected":y,
        "selection_mode":name,
        "batch_cap":cap,
        "batch_fraction":frac,
        "step_mode":mode,
        "routes_considered":len(routes),
        "exact_frontier":False,
    }


def contraction_map(selected):
    """Deterministic survivor map for a disjoint batch."""
    out={}
    for _,r in selected.iterrows():
        u,v=_pair_tuple(r)
        out[v]=u
    return out


def apply_contractions(labels,selected):
    out=np.asarray(labels,dtype=np.int64).copy()
    mp=contraction_map(selected)
    for removed,survivor in mp.items():
        out[out==removed]=survivor
    return out


def summarize_numeric(x,prefix):
    a=np.asarray(x,dtype=np.float64)
    a=a[np.isfinite(a)]
    if len(a)==0:
        return {}
    q=np.quantile(a,[.01,.05,.25,.5,.75,.95,.99])
    return {
        f"{prefix}_n":int(len(a)),
        f"{prefix}_mean":float(np.mean(a)),
        f"{prefix}_sd":float(np.std(a)),
        f"{prefix}_mad":float(np.median(np.abs(a-np.median(a)))),
        f"{prefix}_q01":float(q[0]),
        f"{prefix}_q05":float(q[1]),
        f"{prefix}_q25":float(q[2]),
        f"{prefix}_q50":float(q[3]),
        f"{prefix}_q75":float(q[4]),
        f"{prefix}_q95":float(q[5]),
        f"{prefix}_q99":float(q[6]),
        f"{prefix}_min":float(np.min(a)),
        f"{prefix}_max":float(np.max(a)),
    }


def shannon_from_sizes(sizes):
    x=np.asarray(sizes,dtype=np.float64)
    x=x[x>0]
    if len(x)==0:return 0.0
    p=x/x.sum()
    return float(-(p*np.log(p)).sum())


def flow_landmark_due(removed_fraction,last_landmark,cfg):
    step=cfg.heavy_landmark_reduction_fraction
    if step<=0:
        return False,last_landmark
    target=math.floor(removed_fraction/step)*step
    return (target>last_landmark+1e-15),target
