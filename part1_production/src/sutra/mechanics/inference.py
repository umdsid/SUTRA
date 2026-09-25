from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

@dataclass(frozen=True)
class MechanicsConfig:
    min_confidence_class: str = "admissible"
    lambda_tension: float = 1e-2
    lambda_pressure: float = 1e-2
    lambda_mean_pressure: float = 1e2
    min_interface_length: float = 1e-8

_CONF = {"marginal":0, "admissible":1, "high_confidence":2}

def select_mechanics_edges(df: pd.DataFrame, cfg: MechanicsConfig):
    out = df.copy()
    rank = out["confidence_class"].map(lambda x:_CONF.get(str(x),-1))
    keep = rank >= _CONF[cfg.min_confidence_class]
    keep &= np.isfinite(out["shared_support_at_chosen"].to_numpy(float))
    keep &= out["shared_support_at_chosen"].to_numpy(float) > cfg.min_interface_length
    keep &= np.isfinite(out["normal_i_to_j_x"].to_numpy(float))
    keep &= np.isfinite(out["normal_i_to_j_y"].to_numpy(float))
    return out.loc[keep].reset_index(drop=True)

def build_system(edges: pd.DataFrame, cell_ids: list[str]):
    idx = {c:i for i,c in enumerate(cell_ids)}
    m, n = len(edges), len(cell_ids)
    rows=[]; cols=[]; vals=[]
    for eidx,r in enumerate(edges.itertuples(index=False)):
        i=idx.get(str(r.cell_i)); j=idx.get(str(r.cell_j))
        if i is None or j is None: continue
        nx=float(r.normal_i_to_j_x); ny=float(r.normal_i_to_j_y)
        norm=math.hypot(nx,ny)
        if not np.isfinite(norm) or norm==0: continue
        nx/=norm; ny/=norm
        L=float(r.shared_support_at_chosen)
        for comp,nc in ((0,nx),(1,ny)):
            ri=2*i+comp; rj=2*j+comp
            # tau column
            rows += [ri,rj]; cols += [eidx,eidx]; vals += [nc,-nc]
            # pressure jump
            pi=m+i; pj=m+j
            rows += [ri,ri,rj,rj]
            cols += [pi,pj,pi,pj]
            vals += [-0.5*L*nc, 0.5*L*nc, 0.5*L*nc, -0.5*L*nc]
    A=sparse.coo_matrix((vals,(rows,cols)),shape=(2*n,m+n)).tocsr()
    return A

def solve_mechanics(edges: pd.DataFrame, cell_ids: list[str], cfg: MechanicsConfig):
    A=build_system(edges,cell_ids)
    m,n=len(edges),len(cell_ids)
    blocks=[A]; rhs=[np.zeros(A.shape[0])]
    if m:
        Rt=sparse.hstack([math.sqrt(cfg.lambda_tension)*sparse.eye(m), sparse.csr_matrix((m,n))])
        blocks.append(Rt); rhs.append(np.zeros(m))
    if n:
        Rp=sparse.hstack([sparse.csr_matrix((n,m)), math.sqrt(cfg.lambda_pressure)*sparse.eye(n)])
        blocks.append(Rp); rhs.append(np.zeros(n))
        data=np.full(n, math.sqrt(cfg.lambda_mean_pressure)/max(n,1))
        gauge=sparse.csr_matrix((data,(np.zeros(n,dtype=int),np.arange(m,m+n))),shape=(1,m+n))
        blocks.append(gauge); rhs.append(np.zeros(1))
    Aaug=sparse.vstack(blocks,format="csr")
    baug=np.concatenate(rhs)
    sol=lsqr(Aaug,baug,atol=1e-9,btol=1e-9,iter_lim=max(2000,3*(m+n)))
    x=sol[0]
    resid=A@x
    cell_resid=np.sqrt(resid[0::2]**2+resid[1::2]**2)
    return {
        "tension":x[:m],
        "pressure":x[m:m+n],
        "cell_residual":cell_resid,
        "lsqr_istop":int(sol[1]),
        "lsqr_iterations":int(sol[2]),
        "residual_norm":float(sol[3]),
        "condition_proxy":float(sol[6]),
        "solution_norm":float(sol[8]),
    }

def attach(edges: pd.DataFrame, cell_ids: list[str], sol: dict):
    e=edges.copy()
    e["tension_like"]=sol["tension"]
    p=pd.DataFrame({
        "cell_id":cell_ids,
        "pressure_like":sol["pressure"],
        "force_balance_residual":sol["cell_residual"],
    })
    return e,p
