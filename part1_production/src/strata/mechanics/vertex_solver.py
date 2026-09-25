from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class VertexMechanicsConfig:
    mean_tension_target: float = 1.0
    tension_scale_weight: float = 100.0
    pressure_gauge_weight: float = 100.0
    ridge_tension: float = 1e-4
    ridge_pressure: float = 1e-4
    solver_tol: float = 1e-6
    solver_max_iter: int = 200


def build_vertex_system(edges: pd.DataFrame, cell_ids: list[str], n_junctions: int):
    """
    Discrete junction force balance.

    For interface e=(i,j) with arc length L, tangent t from endpoint 0 to 1,
    and normal n from i to j:

      endpoint 0: +tau_e t + 1/2 (p_i-p_j) L n
      endpoint 1: -tau_e t + 1/2 (p_i-p_j) L n

    Pressure loading is distributed equally to the two interface endpoints.
    """
    cidx={c:k for k,c in enumerate(cell_ids)}
    m=len(edges); n=len(cell_ids)
    rr=[]; cc=[]; vv=[]

    def add(row,col,val):
        rr.append(row); cc.append(col); vv.append(val)

    for eidx,r in enumerate(edges.itertuples(index=False)):
        i=cidx.get(str(r.cell_i)); j=cidx.get(str(r.cell_j))
        if i is None or j is None: continue
        L=float(r.interface_length)
        tx,ty=float(r.tangent_x),float(r.tangent_y)
        nx,ny=float(r.normal_i_to_j_x),float(r.normal_i_to_j_y)
        for endpoint,jid,sgn_t in [(0,int(r.junction0),+1.0),(1,int(r.junction1),-1.0)]:
            for comp,(tc,nc) in enumerate(((tx,nx),(ty,ny))):
                row=2*jid+comp
                add(row,eidx,sgn_t*tc)
                add(row,m+i,+0.5*L*nc)
                add(row,m+j,-0.5*L*nc)

    A=sparse.coo_matrix((vv,(rr,cc)),shape=(2*n_junctions,m+n)).tocsr()
    return A


def solve_vertex_mechanics(edges, cell_ids, n_junctions, cfg: VertexMechanicsConfig):
    A=build_vertex_system(edges,cell_ids,n_junctions)
    m=len(edges); n=len(cell_ids)
    blocks=[A]; rhs=[np.zeros(A.shape[0])]

    # Fix mechanical scale: mean tau = target.
    if m:
        data=np.full(m, cfg.tension_scale_weight/max(m,1))
        scale=sparse.csr_matrix(
            (data,(np.zeros(m,dtype=int),np.arange(m))),
            shape=(1,m+n)
        )
        blocks.append(scale)
        rhs.append(np.array([cfg.tension_scale_weight*cfg.mean_tension_target]))

        rt=sparse.hstack([
            math.sqrt(cfg.ridge_tension)*sparse.eye(m,format="csr"),
            sparse.csr_matrix((m,n))
        ])
        blocks.append(rt); rhs.append(np.zeros(m))

    # Fix pressure gauge: mean p = 0.
    if n:
        data=np.full(n,cfg.pressure_gauge_weight/max(n,1))
        gauge=sparse.csr_matrix(
            (data,(np.zeros(n,dtype=int),np.arange(m,m+n))),
            shape=(1,m+n)
        )
        blocks.append(gauge); rhs.append(np.zeros(1))

        rp=sparse.hstack([
            sparse.csr_matrix((n,m)),
            math.sqrt(cfg.ridge_pressure)*sparse.eye(n,format="csr")
        ])
        blocks.append(rp); rhs.append(np.zeros(n))

    Aaug=sparse.vstack(blocks,format="csr")
    baug=np.concatenate(rhs)

    lower=np.concatenate([np.zeros(m),np.full(n,-np.inf)])
    upper=np.full(m+n,np.inf)

    sol=lsq_linear(
        Aaug,baug,bounds=(lower,upper),
        method="trf",lsq_solver="lsmr",
        tol=cfg.solver_tol,max_iter=cfg.solver_max_iter,
        lsmr_tol="auto",verbose=0
    )
    x=sol.x
    force=A@x
    jr=np.sqrt(force[0::2]**2+force[1::2]**2)

    tau=x[:m]; p=x[m:m+n]
    scale=max(float(np.mean(tau)) if m else 0.0,1e-12)
    normalized_jr=jr/scale

    return {
        "tension":tau,"pressure":p,
        "junction_residual":jr,
        "normalized_junction_residual":normalized_jr,
        "success":bool(sol.success),
        "status_code":int(sol.status),
        "message":str(sol.message),
        "cost":float(sol.cost),
        "optimality":float(sol.optimality),
        "n_iterations":int(sol.nit),
        "mean_tension":float(np.mean(tau)) if m else 0.0,
        "std_tension":float(np.std(tau)) if m else 0.0,
        "mean_pressure":float(np.mean(p)) if n else 0.0,
        "std_pressure":float(np.std(p)) if n else 0.0,
    }
