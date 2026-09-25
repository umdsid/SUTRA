"""STRATA v0.7.6.2.1 pressure-field finalization numerical certificate fix.

The decomposition is constructed algebraically as

    observed = potential + residual,

with residual = observed - potential.

For very large floating-point pressure potentials, reconstructing
potential + residual can lose several absolute ulps through cancellation.
Therefore exactness must be certified with a scale-aware backward-error test,
not a fixed absolute tolerance.

No pressure values, residuals, hierarchy outputs, or transport policy change.
"""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

DP_COLUMNS=("delta_pressure_z","delta_p_z","pressure_difference_z")


def resolve_dp_column(df):
    for c in DP_COLUMNS:
        if c in df.columns:
            return c
    raise RuntimeError(f"no signed delta-p column found; expected {DP_COLUMNS}")


def decompose_pressure_field(edge_rel,n_cells,eps=1e-12):
    dcol=resolve_dp_column(edge_rel)
    i=edge_rel.cell_i_index.to_numpy(np.int64)
    j=edge_rel.cell_j_index.to_numpy(np.int64)
    obs=edge_rel[dcol].to_numpy(np.float64)

    valid=(edge_rel.mechanics_delta_p_valid.astype(bool).to_numpy()
           if "mechanics_delta_p_valid" in edge_rel.columns
           else np.isfinite(obs))
    q=valid & np.isfinite(obs)

    ii=i[q]; jj=j[q]; yy=obs[q]
    rows=np.arange(len(ii),dtype=np.int64)
    B=sparse.coo_matrix(
        (
            np.concatenate([np.ones(len(rows)), -np.ones(len(rows))]),
            (
                np.concatenate([rows,rows]),
                np.concatenate([ii,jj]),
            ),
        ),
        shape=(len(rows),n_cells),
    ).tocsr()

    sol=lsqr(
        B,yy,atol=1e-10,btol=1e-10,
        iter_lim=max(1000,min(20000,5*n_cells)),show=False
    )
    p=sol[0].astype(np.float64)

    potential=np.asarray(B@p,dtype=np.float64)
    residual=yy-potential
    recon=potential+residual
    err=recon-yy

    scale=np.maximum.reduce([
        np.abs(yy),
        np.abs(potential),
        np.abs(residual),
        np.ones_like(yy),
    ])
    # Backward error in units of the local arithmetic scale.
    relerr=np.abs(err)/scale

    chi=np.abs(residual)/(np.abs(yy)+eps)

    E=pd.DataFrame({
        "source_edge_row":np.flatnonzero(q).astype(np.int64),
        "cell_i_index":ii,
        "cell_j_index":jj,
        "delta_p_observed":yy,
        "pressure_i":p[ii],
        "pressure_j":p[jj],
        "delta_p_potential":potential,
        "delta_p_residual":residual,
        "delta_p_reconstructed":recon,
        "reconstruction_error":err,
        "reconstruction_scale":scale,
        "reconstruction_relative_error":relerr,
        "abs_delta_p_residual":np.abs(residual),
        "pressure_consistency_ratio":chi,
    })

    A=sparse.coo_matrix(
        (
            np.ones(2*len(E),dtype=np.int8),
            (
                np.concatenate([ii,jj]),
                np.concatenate([jj,ii]),
            ),
        ),
        shape=(n_cells,n_cells),
    ).tocsr()
    ncomp,labels=connected_components(A,directed=False,return_labels=True)
    E["pressure_component"]=labels[ii]

    cell=pd.DataFrame({
        "cell_index":np.arange(n_cells,dtype=np.int64),
        "pressure_potential":p,
        "pressure_component":labels.astype(np.int64),
    })

    solver={
        "lsqr_istop":int(sol[1]),
        "lsqr_iterations":int(sol[2]),
        "lsqr_r1norm":float(sol[3]),
        "lsqr_r2norm":float(sol[4]),
        "lsqr_anorm":float(sol[5]),
        "lsqr_acond":float(sol[6]),
        "lsqr_arnorm":float(sol[7]),
        "lsqr_xnorm":float(sol[8]),
    }
    return cell,E,solver


def decomposition_summary(E,relative_tol=64*np.finfo(np.float64).eps):
    obs=E.delta_p_observed.to_numpy(np.float64)
    pot=E.delta_p_potential.to_numpy(np.float64)
    res=E.delta_p_residual.to_numpy(np.float64)
    err=E.reconstruction_error.to_numpy(np.float64)
    rel=E.reconstruction_relative_error.to_numpy(np.float64)
    chi=E.pressure_consistency_ratio.to_numpy(np.float64)
    absr=np.abs(res)

    max_abs=float(np.max(np.abs(err)))
    max_rel=float(np.max(rel))
    rms=float(np.sqrt(np.mean(err*err)))

    return {
        "n_constraints":int(len(E)),
        "n_components":int(E.pressure_component.nunique()),
        "reconstruction_max_abs_error":max_abs,
        "reconstruction_rms_error":rms,
        "reconstruction_max_relative_error":max_rel,
        "reconstruction_relative_tolerance":float(relative_tol),
        "reconstruction_max_error_in_eps_units":float(
            max_rel/np.finfo(np.float64).eps
        ),
        "residual_median_abs":float(np.median(absr)),
        "residual_q95_abs":float(np.quantile(absr,.95)),
        "residual_q99_abs":float(np.quantile(absr,.99)),
        "residual_q999_abs":float(np.quantile(absr,.999)),
        "residual_max_abs":float(np.max(absr)),
        "residual_rms":float(np.sqrt(np.mean(res*res))),
        "potential_rms":float(np.sqrt(np.mean(pot*pot))),
        "observed_rms":float(np.sqrt(np.mean(obs*obs))),
        "consistency_ratio_median":float(np.median(chi)),
        "consistency_ratio_q95":float(np.quantile(chi,.95)),
        "consistency_ratio_q99":float(np.quantile(chi,.99)),
        "all_finite":bool(np.isfinite(E[
            [
                "delta_p_observed","delta_p_potential","delta_p_residual",
                "delta_p_reconstructed","reconstruction_error",
                "reconstruction_relative_error","pressure_consistency_ratio",
            ]
        ].to_numpy()).all()),
        "exact_additive_decomposition_float64":bool(max_rel<=relative_tol),
    }


def component_summary(E):
    rows=[]
    for comp,g in E.groupby("pressure_component",sort=False):
        r=g.delta_p_residual.to_numpy(np.float64)
        p=g.delta_p_potential.to_numpy(np.float64)
        o=g.delta_p_observed.to_numpy(np.float64)
        nodes=np.unique(np.concatenate([
            g.cell_i_index.to_numpy(np.int64),
            g.cell_j_index.to_numpy(np.int64),
        ]))
        rows.append({
            "pressure_component":int(comp),
            "n_nodes":int(len(nodes)),
            "n_edges":int(len(g)),
            "residual_median_abs":float(np.median(np.abs(r))),
            "residual_q95_abs":float(np.quantile(np.abs(r),.95)),
            "residual_max_abs":float(np.max(np.abs(r))),
            "residual_rms":float(np.sqrt(np.mean(r*r))),
            "potential_rms":float(np.sqrt(np.mean(p*p))),
            "observed_rms":float(np.sqrt(np.mean(o*o))),
            "consistency_ratio_median":float(np.median(g.pressure_consistency_ratio)),
            "consistency_ratio_q95":float(np.quantile(g.pressure_consistency_ratio,.95)),
        })
    return pd.DataFrame(rows)


def array_sha256(x):
    a=np.asarray(x)
    h=hashlib.sha256()
    h.update(np.asarray(a.shape,dtype=np.int64).tobytes())
    h.update(a.astype(np.float64,copy=False).tobytes(order="C"))
    return h.hexdigest()
