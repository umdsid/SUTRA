from __future__ import annotations
from dataclasses import dataclass
import math, time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

from strata.mechanics.vertex_solver_v2 import build_vertex_system
from strata.mechanics.regularized_solver import build_tension_coupling


@dataclass
class CachedSystem:
    A: sparse.csr_matrix
    L: sparse.csr_matrix
    n_edges: int
    n_cells: int
    scale: sparse.csr_matrix
    pressure_ridge: sparse.csr_matrix
    gauge: sparse.csr_matrix
    coupling_normalization: float


def _fro_norm_sq(M):
    if M.nnz == 0:
        return 0.0
    return float(np.dot(M.data, M.data))


def build_cached_system(edges: pd.DataFrame, cell_ids: list[str], n_junctions: int, ridge_pressure=1e-6):
    A=build_vertex_system(edges,cell_ids,n_junctions).tocsr()
    L=build_tension_coupling(edges).tocsr()
    m=len(edges); n=len(cell_ids)

    # Compare only the tension columns of equilibrium to L.
    At=A[:, :m]
    a2=_fro_norm_sq(At)
    l2=_fro_norm_sq(L)

    # Rescale L so ||L_norm||_F ~= ||A_tau||_F.
    # Then lambda is dimensionless at the operator level.
    norm_factor=math.sqrt(a2/l2) if l2>0 and a2>0 else 1.0
    L=(norm_factor*L).tocsr()

    scale=sparse.csr_matrix(
        (np.full(m,10.0/max(m,1)),(np.zeros(m,dtype=int),np.arange(m))),
        shape=(1,m+n)
    )

    pressure_ridge=sparse.hstack([
        sparse.csr_matrix((n,m)),
        math.sqrt(ridge_pressure)*sparse.eye(n,format="csr")
    ],format="csr")

    gauge=sparse.csr_matrix(
        (np.full(n,10.0/max(n,1)),(np.zeros(n,dtype=int),np.arange(m,m+n))),
        shape=(1,m+n)
    )

    return CachedSystem(
        A=A,L=L,n_edges=m,n_cells=n,scale=scale,
        pressure_ridge=pressure_ridge,gauge=gauge,
        coupling_normalization=float(norm_factor),
    )


def solve_cached(cache: CachedSystem, lam: float, max_iter=1000, tol=1e-5):
    m,n=cache.n_edges,cache.n_cells
    blocks=[cache.A,cache.scale]
    rhs=[np.zeros(cache.A.shape[0]),np.array([10.0])]

    if lam>0 and cache.L.shape[0]>0:
        Lt=sparse.hstack([
            math.sqrt(lam)*cache.L,
            sparse.csr_matrix((cache.L.shape[0],n))
        ],format="csr")
        blocks.append(Lt); rhs.append(np.zeros(cache.L.shape[0]))

    blocks.extend([cache.pressure_ridge,cache.gauge])
    rhs.extend([np.zeros(n),np.zeros(1)])

    Aaug=sparse.vstack(blocks,format="csr")
    baug=np.concatenate(rhs)

    lo=np.r_[np.zeros(m),np.full(n,-np.inf)]
    hi=np.full(m+n,np.inf)

    t0=time.time()
    opt=lsq_linear(
        Aaug,baug,bounds=(lo,hi),method="trf",lsq_solver="lsmr",
        tol=tol,max_iter=max_iter,lsmr_tol="auto",verbose=0
    )

    tau=opt.x[:m]
    p=opt.x[m:m+n]
    mt=float(np.mean(tau))
    if mt<=1e-12:
        raise RuntimeError("vanishing tension scale")
    tau=tau/mt
    p=p/mt
    p=p-float(np.mean(p))

    x=np.r_[tau,p]
    force=cache.A@x
    residual=np.sqrt(force[0::2]**2+force[1::2]**2)

    return {
        "tension":tau,
        "pressure":p,
        "junction_residual":residual,
        "success":bool(opt.success),
        "optimality":float(opt.optimality),
        "n_iterations":int(opt.nit),
        "runtime_seconds":float(time.time()-t0),
    }


def _neff(t):
    return float(np.sum(t)**2/(np.sum(t*t)+1e-15))

def _topmass(t,frac):
    n=len(t); k=max(1,int(np.ceil(frac*n)))
    idx=np.argpartition(t,-k)[-k:]
    return float(np.sum(t[idx])/(np.sum(t)+1e-15))

def _rho(a,b):
    r=spearmanr(a,b).statistic
    return float(r) if np.isfinite(r) else np.nan


def evaluate_solution(lam,sol,base_tau,bmed,bq95,n_edges):
    t=sol["tension"]; r=sol["junction_residual"]
    return {
        "lambda_tau":float(lam),
        "neff_fraction":_neff(t)/max(n_edges,1),
        "top1_mass":_topmass(t,0.01),
        "top5_mass":_topmass(t,0.05),
        "tension_spearman_vs_unregularized":_rho(base_tau,t),
        "median_residual":float(np.median(r)),
        "q95_residual":float(np.quantile(r,0.95)),
        "median_residual_ratio":float(np.median(r)/(bmed+1e-15)),
        "q95_residual_ratio":float(np.quantile(r,0.95)/(bq95+1e-15)),
        "optimizer_success":sol["success"],
        "optimizer_optimality":sol["optimality"],
        "optimizer_iterations":sol["n_iterations"],
        "runtime_seconds":sol["runtime_seconds"],
    }


def adaptive_calibration(edges,cell_ids,n_junctions,base_tau,base_resid):
    """
    Operator-normalized coarse-to-fine sweep.

    Lambda is dimensionless because the tension-coupling operator is rescaled
    to the Frobenius norm of the tension block of the equilibrium operator.

    Start much smaller than before. Abort an upward branch once residual
    degradation is catastrophic and monotone.
    """
    cache=build_cached_system(edges,cell_ids,n_junctions)
    bmed=float(np.median(base_resid))
    bq95=float(np.quantile(base_resid,0.95))
    n_edges=len(edges)

    print(f"      coupling operator normalization = {cache.coupling_normalization:.3e}", flush=True)

    tested={}
    solutions={}

    def run_one(lam):
        lam=float(lam)
        if lam in tested:
            return tested[lam]
        print(f"        solving lambda={lam:.1e} ...",flush=True)
        sol=solve_cached(cache,lam)
        row=evaluate_solution(lam,sol,base_tau,bmed,bq95,n_edges)
        tested[lam]=row
        solutions[lam]=sol
        print(
            f"          done: Neff={100*row['neff_fraction']:.2f}% "
            f"top1={100*row['top1_mass']:.1f}% "
            f"rho={row['tension_spearman_vs_unregularized']:.3f} "
            f"q95x={row['q95_residual_ratio']:.3f} "
            f"{row['runtime_seconds']:.1f}s",
            flush=True
        )
        return row

    # Dimensionless coarse sweep after operator normalization.
    coarse=[1e-8,1e-7,1e-6,1e-5,1e-4]

    bad_streak=0
    for lam in coarse:
        row=run_one(lam)
        if row["q95_residual_ratio"]>5.0:
            bad_streak += 1
        else:
            bad_streak = 0

        # If two successive larger lambdas are catastrophically bad, do not
        # waste time going upward.
        if bad_streak>=2:
            print("        early stop: two consecutive q95 residual ratios > 5", flush=True)
            break

    df=pd.DataFrame(tested.values()).sort_values("lambda_tau").reset_index(drop=True)

    admiss=df[
        (df.q95_residual_ratio<=1.25) &
        (df.tension_spearman_vs_unregularized>=0.85)
    ].copy()

    if admiss.empty:
        return df,None,None,cache

    # Candidate maximizing concentration relief while preserving admissibility.
    score=admiss.neff_fraction-admiss.top1_mass
    best=float(admiss.loc[score.idxmax(),"lambda_tau"])

    # One or two log-neighbor refinements only.
    grid=[1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4]
    if best in grid:
        pos=grid.index(best)
        candidates=[]
        if pos>0: candidates.append(grid[pos-1])
        if pos<len(grid)-1: candidates.append(grid[pos+1])
        for lam in candidates[:2]:
            run_one(lam)

    df=pd.DataFrame(tested.values()).sort_values("lambda_tau").reset_index(drop=True)
    admiss=df[
        (df.q95_residual_ratio<=1.25) &
        (df.tension_spearman_vs_unregularized>=0.85)
    ].copy()
    if admiss.empty:
        return df,None,None,cache

    score=admiss.neff_fraction-admiss.top1_mass
    chosen=float(admiss.loc[score.idxmax(),"lambda_tau"])
    return df,chosen,solutions[chosen],cache
