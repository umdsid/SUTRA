"""STRATA v0.7.5 Step 2: raw node-local directional covector.

For channel r=(sender s, receiver t) on supported measured contact i--j,

    delta_r(i,j) = f_r(i->j) - f_r(j->i).

Node i receives
    + delta_r (e_t - e_s),

and node j receives
    - delta_r (e_t - e_s).

Thus storage reversal i<->j flips both delta and node role and leaves the
physical node-local covector unchanged.

Each node is divided by its total supported incident channel flux so the raw
covector is dimensionless. Mechanics coordinates receive zero directional
components in Step 2.

No scaling to satisfy the strict Randers/Finsler bound is performed here.
"""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cholesky, solve_triangular


def build_raw_node_covectors(
    Y: np.ndarray,
    edges: pd.DataFrame,
    genes: tuple[str,...],
    registry: pd.DataFrame,
    support_floor: float,
    eps: float=1e-12,
):
    idx={str(g):k for k,g in enumerate(genes)}
    reg=registry.copy()
    if "panel_supported" in reg.columns:
        reg=reg[reg.panel_supported.astype(bool)]
    reg=reg[
        reg.sender_gene.astype(str).isin(idx)
        & reg.receiver_gene.astype(str).isin(idx)
        & reg.kind.astype(str).isin(["contact","diffusible"])
    ].copy()

    i=edges.cell_i_index.to_numpy(np.int64)
    j=edges.cell_j_index.to_numpy(np.int64)
    scalar_support=(
        edges.comm_ij.to_numpy(np.float64)
        + edges.comm_ji.to_numpy(np.float64)
    )
    supported=np.isfinite(scalar_support) & (scalar_support>=support_floor)

    n_nodes=Y.shape[0]
    n_genes=len(genes)
    B=np.zeros((n_nodes,n_genes),dtype=np.float64)
    node_support=np.zeros(n_nodes,dtype=np.float64)
    node_abs_directional_flux=np.zeros(n_nodes,dtype=np.float64)

    kind_support={
        "contact":np.zeros(n_nodes,dtype=np.float64),
        "diffusible":np.zeros(n_nodes,dtype=np.float64),
    }
    kind_abs_direction={
        "contact":np.zeros(n_nodes,dtype=np.float64),
        "diffusible":np.zeros(n_nodes,dtype=np.float64),
    }

    interaction_rows=[]
    for ridx,r in reg.reset_index(drop=True).iterrows():
        s=idx[str(r.sender_gene)]
        t=idx[str(r.receiver_gene)]
        fij=np.sqrt(
            np.maximum(0.0,Y[i,s].astype(np.float64))
            * np.maximum(0.0,Y[j,t].astype(np.float64))
        )
        fji=np.sqrt(
            np.maximum(0.0,Y[j,s].astype(np.float64))
            * np.maximum(0.0,Y[i,t].astype(np.float64))
        )
        delta=fij-fji
        weight=fij+fji

        q=supported
        ii=i[q]; jj=j[q]
        d=delta[q]; w=weight[q]

        # Covector contribution delta * (e_receiver - e_sender)
        np.add.at(B[:,t],ii,d)
        np.add.at(B[:,s],ii,-d)
        np.add.at(B[:,t],jj,-d)
        np.add.at(B[:,s],jj,d)

        np.add.at(node_support,ii,w)
        np.add.at(node_support,jj,w)
        np.add.at(node_abs_directional_flux,ii,np.abs(d))
        np.add.at(node_abs_directional_flux,jj,np.abs(d))

        kind=str(r.kind)
        np.add.at(kind_support[kind],ii,w)
        np.add.at(kind_support[kind],jj,w)
        np.add.at(kind_abs_direction[kind],ii,np.abs(d))
        np.add.at(kind_abs_direction[kind],jj,np.abs(d))

        interaction_rows.append({
            "channel_index":int(ridx),
            "sender_gene":str(r.sender_gene),
            "receiver_gene":str(r.receiver_gene),
            "kind":kind,
            "n_supported_spatial_edges":int(q.sum()),
            "total_supported_channel_flux":float(np.sum(w)),
            "total_supported_abs_directional_flux":float(np.sum(np.abs(d))),
        })

    qnode=node_support>eps
    B[qnode,:]/=node_support[qnode,None]
    B[~qnode,:]=0.0

    # Append mechanics zeros.
    Z=np.zeros((n_nodes,2),dtype=np.float64)
    Bfull=np.hstack([B,Z])
    Bcsr=sparse.csr_matrix(Bfull)
    Bcsr.eliminate_zeros()

    stats=pd.DataFrame({
        "cell_index":np.arange(n_nodes,dtype=np.int64),
        "comm_incident_support":node_support,
        "comm_has_supported_incident_flux":qnode,
        "comm_abs_directional_flux":node_abs_directional_flux,
        "comm_directional_content_local":np.divide(
            node_abs_directional_flux,node_support,
            out=np.full(n_nodes,np.nan,dtype=np.float64),
            where=node_support>eps,
        ),
        "contact_incident_support":kind_support["contact"],
        "contact_abs_directional_flux":kind_abs_direction["contact"],
        "diffusible_incident_support":kind_support["diffusible"],
        "diffusible_abs_directional_flux":kind_abs_direction["diffusible"],
        "raw_covector_l2":np.sqrt(Bcsr.multiply(Bcsr).sum(axis=1)).A1,
        "raw_covector_nnz":np.diff(Bcsr.indptr),
    })
    return Bcsr,stats,pd.DataFrame(interaction_rows)


def dual_norms_spd(G, B, chunk_size=2048):
    """Compute sqrt(b^T G^{-1} b) for rows b without forming G^{-1}."""
    G=np.asarray(G,dtype=np.float64)
    L=cholesky(G,lower=True,check_finite=True)
    B=sparse.csr_matrix(B,dtype=np.float64)
    out=np.zeros(B.shape[0],dtype=np.float64)

    for start in range(0,B.shape[0],chunk_size):
        stop=min(B.shape[0],start+chunk_size)
        X=B[start:stop].toarray().T
        # G = L L^T; b^T G^-1 b = ||L^-1 b||^2
        Y=solve_triangular(L,X,lower=True,check_finite=False)
        out[start:stop]=np.sqrt(np.sum(Y*Y,axis=0))
    return out


def sparse_sha256(A):
    A=sparse.csr_matrix(A,dtype=np.float64)
    h=hashlib.sha256()
    for x in (
        np.asarray(A.shape,dtype=np.int64),
        A.indptr.astype(np.int64,copy=False),
        A.indices.astype(np.int64,copy=False),
        A.data.astype(np.float64,copy=False),
    ):
        h.update(x.tobytes(order="C"))
    return h.hexdigest()


def summarize_covectors(stats,dual):
    active=stats.comm_has_supported_incident_flux.to_numpy(bool)
    d=np.asarray(dual,dtype=np.float64)
    da=d[active]
    return {
        "n_nodes":int(len(stats)),
        "n_nodes_with_supported_incident_flux":int(active.sum()),
        "supported_incident_node_fraction":float(active.mean()),
        "n_nonzero_raw_covectors":int((stats.raw_covector_nnz>0).sum()),
        "nonzero_raw_covector_fraction":float((stats.raw_covector_nnz>0).mean()),
        "dual_norm_max":float(np.max(d)) if len(d) else 0.0,
        "dual_norm_supported_median":float(np.median(da)) if len(da) else np.nan,
        "dual_norm_supported_q90":float(np.quantile(da,.90)) if len(da) else np.nan,
        "dual_norm_supported_q95":float(np.quantile(da,.95)) if len(da) else np.nan,
        "dual_norm_supported_q99":float(np.quantile(da,.99)) if len(da) else np.nan,
        "all_dual_norms_finite":bool(np.isfinite(d).all()),
        "mechanics_directional_components_zero":True,
    }


def reverse_edges_for_test(edges):
    e=edges.copy()
    a=e.cell_i_index.copy()
    e["cell_i_index"]=e.cell_j_index.to_numpy()
    e["cell_j_index"]=a.to_numpy()
    a=e.comm_ij.copy()
    e["comm_ij"]=e.comm_ji.to_numpy()
    e["comm_ji"]=a.to_numpy()
    return e
