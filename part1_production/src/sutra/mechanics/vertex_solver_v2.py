from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class SolverConfig:
    ridge_tension: float = 1e-6
    ridge_pressure: float = 1e-6
    soft_scale_weight: float = 10.0
    soft_pressure_gauge_weight: float = 10.0
    max_iter: int = 1000
    tol: float = 1e-5


def build_vertex_system(edges: pd.DataFrame, cell_ids: list[str], n_junctions: int):
    idx={c:k for k,c in enumerate(cell_ids)}
    m,n=len(edges),len(cell_ids)
    rr=[];cc=[];vv=[]
    def add(r,c,v):
        rr.append(r);cc.append(c);vv.append(v)
    for eidx,r in enumerate(edges.itertuples(index=False)):
        i=idx.get(str(r.cell_i)); j=idx.get(str(r.cell_j))
        if i is None or j is None: continue
        L=float(r.interface_length)
        tx,ty=float(r.tangent_x),float(r.tangent_y)
        nx,ny=float(r.normal_i_to_j_x),float(r.normal_i_to_j_y)
        for jid,sgn in ((int(r.junction0),1.0),(int(r.junction1),-1.0)):
            for comp,(tc,nc) in enumerate(((tx,nx),(ty,ny))):
                row=2*jid+comp
                add(row,eidx,sgn*tc)
                add(row,m+i,+0.5*L*nc)
                add(row,m+j,-0.5*L*nc)
    return sparse.coo_matrix((vv,(rr,cc)),shape=(2*n_junctions,m+n)).tocsr()


def solve(edges,cell_ids,n_junctions,cfg=SolverConfig()):
    A=build_vertex_system(edges,cell_ids,n_junctions)
    m,n=len(edges),len(cell_ids)

    blocks=[A]; rhs=[np.zeros(A.shape[0])]

    # Soft constraints only select a nonzero representative before exact gauge
    # normalization. Exact scale/gauge are imposed after solve, which is valid
    # because the equilibrium equations are homogeneous.
    if m:
        scale=sparse.csr_matrix(
            (np.full(m,cfg.soft_scale_weight/max(m,1)),
             (np.zeros(m,dtype=int),np.arange(m))),
            shape=(1,m+n)
        )
        blocks.append(scale);rhs.append(np.array([cfg.soft_scale_weight]))
        rt=sparse.hstack([
            np.sqrt(cfg.ridge_tension)*sparse.eye(m,format="csr"),
            sparse.csr_matrix((m,n))
        ])
        blocks.append(rt);rhs.append(np.zeros(m))
    if n:
        gauge=sparse.csr_matrix(
            (np.full(n,cfg.soft_pressure_gauge_weight/max(n,1)),
             (np.zeros(n,dtype=int),np.arange(m,m+n))),
            shape=(1,m+n)
        )
        blocks.append(gauge);rhs.append(np.zeros(1))
        rp=sparse.hstack([
            sparse.csr_matrix((n,m)),
            np.sqrt(cfg.ridge_pressure)*sparse.eye(n,format="csr")
        ])
        blocks.append(rp);rhs.append(np.zeros(n))

    Aaug=sparse.vstack(blocks,format="csr")
    baug=np.concatenate(rhs)
    lower=np.concatenate([np.zeros(m),np.full(n,-np.inf)])
    upper=np.full(m+n,np.inf)

    opt=lsq_linear(
        Aaug,baug,bounds=(lower,upper),method="trf",lsq_solver="lsmr",
        tol=cfg.tol,max_iter=cfg.max_iter,lsmr_tol="auto",verbose=0
    )
    x=opt.x.copy()
    tau=x[:m]; p=x[m:m+n]

    # EXACT mechanical scale and pressure gauge.
    mean_tau=float(np.mean(tau)) if m else 0.0
    if mean_tau<=1e-12:
        raise RuntimeError("mechanical solution has vanishing tension scale")
    tau=tau/mean_tau
    p=p/mean_tau
    p=p-float(np.mean(p))

    x_exact=np.concatenate([tau,p])
    residual=A@x_exact
    jr=np.sqrt(residual[0::2]**2+residual[1::2]**2)

    # Unit-tension / zero-pressure geometric baseline.
    x0=np.concatenate([np.ones(m),np.zeros(n)])
    r0=A@x0
    br=np.sqrt(r0[0::2]**2+r0[1::2]**2)

    return {
        "tension":tau,
        "pressure":p,
        "junction_residual":jr,
        "baseline_junction_residual":br,
        "success":bool(opt.success),
        "status_code":int(opt.status),
        "message":str(opt.message),
        "cost":float(opt.cost),
        "optimality":float(opt.optimality),
        "n_iterations":int(opt.nit),
        "mean_tension":float(np.mean(tau)) if m else 0.0,
        "std_tension":float(np.std(tau)) if m else 0.0,
        "mean_pressure":float(np.mean(p)) if n else 0.0,
        "std_pressure":float(np.std(p)) if n else 0.0,
    }
