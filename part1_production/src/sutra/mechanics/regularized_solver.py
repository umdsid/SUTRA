from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import lsq_linear

from sutra.mechanics.vertex_solver_v2 import build_vertex_system


@dataclass(frozen=True)
class RegularizedConfig:
    lambda_tau: float
    ridge_pressure: float = 1e-6
    max_iter: int = 1200
    tol: float = 1e-5


def build_tension_coupling(edges: pd.DataFrame):
    """
    Geometry-aware graph Laplacian on interface tensions.

    Interfaces are coupled when they share a reconstructed junction.
    Weight = orientation compatibility / sqrt(local multiplicity), where
    compatibility = |t_e dot t_f|. This preserves coherent aligned load paths
    while penalizing isolated amplitude spikes.
    """
    by_j={}
    for eidx,r in enumerate(edges.itertuples(index=False)):
        for jid in (int(r.junction0),int(r.junction1)):
            by_j.setdefault(jid,[]).append(eidx)

    rr=[];cc=[];vv=[]; row=0
    tang=edges[["tangent_x","tangent_y"]].to_numpy(float)
    for jid,inc in by_j.items():
        if len(inc)<2: continue
        mult=max(len(inc)-1,1)
        for a in range(len(inc)):
            for b in range(a+1,len(inc)):
                e,f=inc[a],inc[b]
                w=abs(float(np.dot(tang[e],tang[f])))
                w=max(w,0.1)/math.sqrt(mult)
                s=math.sqrt(w)
                rr += [row,row]; cc += [e,f]; vv += [s,-s]
                row += 1
    return sparse.coo_matrix((vv,(rr,cc)),shape=(row,len(edges))).tocsr()


def solve_regularized(edges,cell_ids,n_junctions,cfg:RegularizedConfig):
    A=build_vertex_system(edges,cell_ids,n_junctions)
    m,n=len(edges),len(cell_ids)
    L=build_tension_coupling(edges)

    blocks=[A]; rhs=[np.zeros(A.shape[0])]

    # Nonzero representative. Exact scale is imposed after solve.
    scale=sparse.csr_matrix(
        (np.full(m,10.0/max(m,1)),(np.zeros(m,dtype=int),np.arange(m))),
        shape=(1,m+n)
    )
    blocks.append(scale); rhs.append(np.array([10.0]))

    if cfg.lambda_tau>0 and L.shape[0]>0:
        Lt=sparse.hstack([math.sqrt(cfg.lambda_tau)*L,sparse.csr_matrix((L.shape[0],n))])
        blocks.append(Lt); rhs.append(np.zeros(L.shape[0]))

    if n:
        rp=sparse.hstack([sparse.csr_matrix((n,m)),math.sqrt(cfg.ridge_pressure)*sparse.eye(n,format="csr")])
        blocks.append(rp);rhs.append(np.zeros(n))
        gauge=sparse.csr_matrix(
            (np.full(n,10.0/max(n,1)),(np.zeros(n,dtype=int),np.arange(m,m+n))),
            shape=(1,m+n)
        )
        blocks.append(gauge);rhs.append(np.zeros(1))

    Aaug=sparse.vstack(blocks,format="csr")
    baug=np.concatenate(rhs)
    lo=np.r_[np.zeros(m),np.full(n,-np.inf)]
    hi=np.full(m+n,np.inf)

    opt=lsq_linear(
        Aaug,baug,bounds=(lo,hi),method="trf",lsq_solver="lsmr",
        tol=cfg.tol,max_iter=cfg.max_iter,lsmr_tol="auto",verbose=0
    )
    tau=opt.x[:m]; p=opt.x[m:m+n]
    mt=float(np.mean(tau))
    if mt<=1e-12: raise RuntimeError("vanishing tension scale")
    tau=tau/mt
    p=p/mt
    p=p-float(np.mean(p))

    x=np.r_[tau,p]
    force=A@x
    jr=np.sqrt(force[0::2]**2+force[1::2]**2)
    return {
        "tension":tau,"pressure":p,"junction_residual":jr,
        "success":bool(opt.success),"optimality":float(opt.optimality),
        "n_iterations":int(opt.nit),
        "lambda_tau":float(cfg.lambda_tau),
    }
