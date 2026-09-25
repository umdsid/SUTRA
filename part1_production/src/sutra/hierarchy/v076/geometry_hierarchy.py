"""Geometry-aware hierarchy core for STRATA v0.7.6.

Biological admissibility is inherited unchanged from v0.7.4.2.
The certified local direction-dependent geometry is used only to order
already-admissible contractions.

For current supernodes A,B with effective state displacement

    v_AB = z_B - z_A,

the orientation-independent pair cost is

    C_AB = 1/2 [ F_A(v_AB) + F_B(-v_AB) ].

Swapping A and B leaves C_AB exactly unchanged, while the directional
correction remains

    1/2 (b_A - b_B)^T v_AB.

The effective state z contains:
  - all measured molecular coordinates;
  - a node/supernode tension coordinate;
  - a pressure-potential coordinate reconstructed from certified signed
    pressure-difference constraints.

No biology gate is replaced by the geometric cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr


TENSION_COLUMNS=("tension_z","tau_z")
DP_COLUMNS=("delta_pressure_z","delta_p_z","pressure_difference_z")


def resolve_column(df,names):
    for x in names:
        if x in df.columns:
            return x
    return None


def reconstruct_level0_mechanics(edge_rel,n_cells):
    """Build Level-0 cell mechanical coordinates from certified interfaces."""
    i=edge_rel.cell_i_index.to_numpy(np.int64)
    j=edge_rel.cell_j_index.to_numpy(np.int64)

    tcol=resolve_column(edge_rel,TENSION_COLUMNS)
    dcol=resolve_column(edge_rel,DP_COLUMNS)
    if tcol is None:
        raise RuntimeError(
            "No signed/standardized tension column found. Expected one of "
            f"{TENSION_COLUMNS}"
        )
    if dcol is None:
        raise RuntimeError(
            "No signed standardized pressure-difference column found. "
            f"Expected one of {DP_COLUMNS}"
        )

    tv=(
        edge_rel.mechanics_tension_valid.astype(bool).to_numpy()
        if "mechanics_tension_valid" in edge_rel.columns
        else np.isfinite(edge_rel[tcol].to_numpy(np.float64))
    )
    dv=(
        edge_rel.mechanics_delta_p_valid.astype(bool).to_numpy()
        if "mechanics_delta_p_valid" in edge_rel.columns
        else np.isfinite(edge_rel[dcol].to_numpy(np.float64))
    )

    tau=edge_rel[tcol].to_numpy(np.float64)
    dp=edge_rel[dcol].to_numpy(np.float64)

    # Node tension: mean of incident certified interface tensions.
    tsum=np.zeros(n_cells,dtype=np.float64)
    tcnt=np.zeros(n_cells,dtype=np.int64)
    q=tv & np.isfinite(tau)
    np.add.at(tsum,i[q],tau[q]); np.add.at(tsum,j[q],tau[q])
    np.add.at(tcnt,i[q],1); np.add.at(tcnt,j[q],1)
    tau_node=np.full(n_cells,np.nan,dtype=np.float64)
    ok=tcnt>0
    tau_node[ok]=tsum[ok]/tcnt[ok]

    # Pressure potential from B p = delta_p with minimum-norm gauge.
    q=dv & np.isfinite(dp)
    rows=np.arange(int(q.sum()),dtype=np.int64)
    ii=i[q]; jj=j[q]
    B=sparse.coo_matrix(
        (
            np.concatenate([
                np.ones(len(rows),dtype=np.float64),
                -np.ones(len(rows),dtype=np.float64),
            ]),
            (
                np.concatenate([rows,rows]),
                np.concatenate([ii,jj]),
            ),
        ),
        shape=(len(rows),n_cells),
    ).tocsr()

    if B.shape[0]==0:
        raise RuntimeError("No certified signed pressure-difference constraints")

    sol=lsqr(
        B,dp[q],
        atol=1e-10,btol=1e-10,
        iter_lim=max(1000,min(20000,5*n_cells)),
        show=False,
    )
    p=sol[0].astype(np.float64)

    pcnt=np.zeros(n_cells,dtype=np.int64)
    np.add.at(pcnt,ii,1); np.add.at(pcnt,jj,1)
    p_obs=pcnt>0

    residual=B@p-dp[q]
    pressure_audit={
        "n_constraints":int(B.shape[0]),
        "n_cells_with_pressure_constraint":int(p_obs.sum()),
        "pressure_constraint_cell_fraction":float(p_obs.mean()),
        "lsqr_istop":int(sol[1]),
        "lsqr_iterations":int(sol[2]),
        "residual_rms":float(np.sqrt(np.mean(residual**2))),
        "residual_median_abs":float(np.median(np.abs(residual))),
        "residual_q95_abs":float(np.quantile(np.abs(residual),.95)),
    }

    stats=pd.DataFrame({
        "cell_index":np.arange(n_cells,dtype=np.int64),
        "tension_state":tau_node,
        "tension_observed":ok,
        "pressure_state":p,
        "pressure_constrained":p_obs,
        "tension_incident_count":tcnt,
        "pressure_constraint_count":pcnt,
    })
    return stats,pressure_audit


def aggregate_mechanics(mech,labels,node_ids):
    out=np.full((len(node_ids),2),np.nan,dtype=np.float64)
    idx={int(x):k for k,x in enumerate(node_ids)}
    for node in node_ids:
        k=idx[int(node)]
        q=np.flatnonzero(labels==node)
        tx=mech.tension_state.to_numpy(np.float64)[q]
        to=mech.tension_observed.to_numpy(bool)[q]
        px=mech.pressure_state.to_numpy(np.float64)[q]
        po=mech.pressure_constrained.to_numpy(bool)[q]
        if np.any(to):
            out[k,0]=float(np.mean(tx[to]))
        if np.any(po):
            out[k,1]=float(np.mean(px[po]))
    return out


def aggregate_covectors(B,labels,node_ids):
    """Cell-count weighted mean covector; preserves strict dual bound by convexity."""
    B=sparse.csr_matrix(B,dtype=np.float64)
    rows=[]
    for node in node_ids:
        q=np.flatnonzero(labels==node)
        if len(q)==1:
            rows.append(B.getrow(q[0]))
        else:
            rows.append(sparse.csr_matrix(B[q].mean(axis=0)))
    return sparse.vstack(rows,format="csr")


def full_effective_states(molecular_states,mechanics_states):
    if molecular_states.shape[0]!=mechanics_states.shape[0]:
        raise ValueError("molecular/mechanics row mismatch")
    return np.hstack([molecular_states.astype(np.float64),mechanics_states])


def geometry_costs(cand,node_ids,states,Bnode,G):
    """Attach orientation-invariant local geometry costs to candidate boundaries."""
    if len(cand)==0:
        return cand.copy()
    idx={int(x):k for k,x in enumerate(node_ids)}
    rows=[]
    for r in cand.itertuples(index=False):
        a=int(r.super_i); b=int(r.super_j)
        ia=idx[a]; ib=idx[b]
        v=states[ib]-states[ia]

        # Missing mechanics state means the geometric state is unresolved.
        finite=bool(np.isfinite(v).all())
        rec=r._asdict()
        rec["geometry_state_resolved"]=finite

        if not finite:
            rec.update({
                "alpha_cost":np.nan,
                "beta_i_forward":np.nan,
                "beta_j_reverse":np.nan,
                "directional_correction":np.nan,
                "geometry_pair_cost":np.nan,
                "local_forward_cost":np.nan,
                "local_reverse_cost":np.nan,
                "geometry_reversal_ratio":np.nan,
            })
            rows.append(rec)
            continue

        alpha=float(np.sqrt(max(0.0,v@(G@v))))
        bi=Bnode.getrow(ia).toarray().ravel()
        bj=Bnode.getrow(ib).toarray().ravel()
        beta_i=float(bi@v)
        beta_j_reverse=float(bj@(-v))
        fi=alpha+beta_i
        fj=alpha+beta_j_reverse
        pair=.5*(fi+fj)
        corr=pair-alpha

        rec.update({
            "alpha_cost":alpha,
            "beta_i_forward":beta_i,
            "beta_j_reverse":beta_j_reverse,
            "directional_correction":corr,
            "geometry_pair_cost":pair,
            "local_forward_cost":fi,
            "local_reverse_cost":fj,
            "geometry_reversal_ratio":(
                max(fi/fj,fj/fi) if fi>0 and fj>0 else np.inf
            ),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def geometry_matching(cand):
    """Deterministic maximal matching ordered by geometry among admissible edges."""
    if len(cand)==0:
        z=cand.copy(); z["selected_geometry"]=False; return z

    a=cand[
        cand.admissible.astype(bool)
        & cand.geometry_state_resolved.astype(bool)
        & np.isfinite(cand.geometry_pair_cost)
    ].copy()

    if len(a)==0:
        a["selected_geometry"]=False
        return a

    a=a.sort_values(
        [
            "geometry_pair_cost",
            "alpha_cost",
            "n_boundary_edges",
            "super_i","super_j",
        ],
        ascending=[True,True,False,True,True],
        kind="mergesort",
    ).reset_index(drop=True)

    used=set(); sel=[]
    for r in a.itertuples():
        u=int(r.super_i); v=int(r.super_j)
        take=(u not in used and v not in used)
        sel.append(take)
        if take:
            used.add(u); used.add(v)
    a["selected_geometry"]=sel
    a["selected"]=a["selected_geometry"]
    return a


def baseline_matching(cand):
    """v0.7.4.2-style deterministic matching for same-level comparison only."""
    if len(cand)==0:
        z=cand.copy(); z["selected_baseline"]=False; return z
    a=cand[cand.admissible.astype(bool)].copy()
    if len(a)==0:
        a["selected_baseline"]=False; return a
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
    a["selected_baseline"]=sel
    return a


def matching_overlap(geom,base):
    G=set(
        zip(
            geom.loc[geom.selected_geometry,"super_i"].astype(int),
            geom.loc[geom.selected_geometry,"super_j"].astype(int),
        )
    ) if len(geom) else set()
    B=set(
        zip(
            base.loc[base.selected_baseline,"super_i"].astype(int),
            base.loc[base.selected_baseline,"super_j"].astype(int),
        )
    ) if len(base) else set()

    inter=len(G&B)
    union=len(G|B)
    return {
        "geometry_selected":len(G),
        "baseline_selected":len(B),
        "selected_intersection":inter,
        "selected_jaccard":float(inter/union) if union else 1.0,
        "geometry_fraction_shared_with_baseline":float(inter/len(G)) if G else 1.0,
    }


def contract_labels(labels,selected):
    out=labels.copy()
    if len(selected)==0:
        return out
    for r in selected[selected.selected_geometry].itertuples():
        a=int(r.super_i);b=int(r.super_j)
        survivor=min(a,b);removed=max(a,b)
        out[out==removed]=survivor
    return out
