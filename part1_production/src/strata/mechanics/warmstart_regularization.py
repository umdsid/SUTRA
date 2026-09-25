from __future__ import annotations
from dataclasses import dataclass
import math, time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import spearmanr

from strata.mechanics.vertex_solver_v2 import build_vertex_system
from strata.mechanics.regularized_solver import build_tension_coupling


@dataclass
class WarmCache:
    A: sparse.csr_matrix
    L: sparse.csr_matrix
    m: int
    n: int
    coupling_normalization: float


def _fro2(M):
    return float(np.dot(M.data,M.data)) if M.nnz else 0.0


def build_cache(edges,cell_ids,n_junctions):
    A=build_vertex_system(edges,cell_ids,n_junctions).tocsr()
    L=build_tension_coupling(edges).tocsr()
    m=len(edges); n=len(cell_ids)
    At=A[:,:m]
    a2=_fro2(At); l2=_fro2(L)
    fac=math.sqrt(a2/l2) if a2>0 and l2>0 else 1.0
    L=(fac*L).tocsr()
    return WarmCache(A=A,L=L,m=m,n=n,coupling_normalization=float(fac))


def _objective_grad(x,cache,lam,mu_scale=100.0,mu_gauge=100.0,ridge_p=1e-8):
    m,n=cache.m,cache.n
    tau=x[:m]; p=x[m:]
    Ax=cache.A@x
    val=0.5*float(np.dot(Ax,Ax))
    grad=cache.A.T@Ax

    if lam>0 and cache.L.shape[0]:
        Lt=cache.L@tau
        val += 0.5*lam*float(np.dot(Lt,Lt))
        grad[:m] += lam*(cache.L.T@Lt)

    mt=float(np.mean(tau))
    mp=float(np.mean(p)) if n else 0.0
    ds=mt-1.0
    val += 0.5*mu_scale*ds*ds
    grad[:m] += (mu_scale*ds/max(m,1))

    if n:
        val += 0.5*mu_gauge*mp*mp + 0.5*ridge_p*float(np.dot(p,p))
        grad[m:] += (mu_gauge*mp/max(n,1)) + ridge_p*p

    return val, np.asarray(grad,float)


def solve_warm(cache,lam,base_tau,base_p,maxiter=250,gtol=1e-5):
    m,n=cache.m,cache.n
    x0=np.r_[np.asarray(base_tau,float),np.asarray(base_p,float)]
    bounds=[(0.0,None)]*m + [(None,None)]*n

    t0=time.time()
    res=minimize(
        fun=lambda x:_objective_grad(x,cache,lam),
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter":maxiter,"gtol":gtol,"ftol":1e-12,"maxls":30}
    )
    x=res.x
    tau=x[:m]; p=x[m:]
    mt=float(np.mean(tau))
    if mt<=1e-12:
        raise RuntimeError("vanishing tension scale")
    tau=tau/mt
    p=p/mt
    p=p-float(np.mean(p))

    xx=np.r_[tau,p]
    force=cache.A@xx
    jr=np.sqrt(force[0::2]**2+force[1::2]**2)

    return {
        "tension":tau,"pressure":p,"junction_residual":jr,
        "success":bool(res.success),
        "message":str(res.message),
        "optimality":float(np.linalg.norm(res.jac,np.inf)),
        "n_iterations":int(res.nit),
        "runtime_seconds":float(time.time()-t0),
    }


def _neff(t):
    return float(np.sum(t)**2/(np.sum(t*t)+1e-15))

def _topmass(t,frac):
    n=len(t);k=max(1,int(np.ceil(frac*n)))
    idx=np.argpartition(t,-k)[-k:]
    return float(np.sum(t[idx])/(np.sum(t)+1e-15))

def _rho(a,b):
    r=spearmanr(a,b).statistic
    return float(r) if np.isfinite(r) else np.nan


def adaptive_warm_calibration(edges,cell_ids,n_junctions,base_tau,base_p,base_resid):
    cache=build_cache(edges,cell_ids,n_junctions)
    bmed=float(np.median(base_resid))
    bq95=float(np.quantile(base_resid,0.95))
    N=len(edges)

    print(f"      warm-start cache normalization = {cache.coupling_normalization:.3e}",flush=True)

    # Start where the perturbation is genuinely small and grow until we either
    # improve concentration or lose admissibility.
    lambdas=[1e-10,3e-10,1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,1e-6]
    rows=[]; sols={}
    bad=0

    for lam in lambdas:
        print(f"        warm solve lambda={lam:.1e} ...",flush=True)
        sol=solve_warm(cache,lam,base_tau,base_p)
        t=sol["tension"]; r=sol["junction_residual"]
        row={
            "lambda_tau":lam,
            "neff_fraction":_neff(t)/max(N,1),
            "top1_mass":_topmass(t,0.01),
            "top5_mass":_topmass(t,0.05),
            "tension_spearman_vs_unregularized":_rho(base_tau,t),
            "median_residual_ratio":float(np.median(r)/(bmed+1e-15)),
            "q95_residual_ratio":float(np.quantile(r,0.95)/(bq95+1e-15)),
            "optimizer_success":sol["success"],
            "optimizer_iterations":sol["n_iterations"],
            "runtime_seconds":sol["runtime_seconds"],
        }
        rows.append(row);sols[lam]=sol
        print(
            f"          done: Neff={100*row['neff_fraction']:.2f}% "
            f"top1={100*row['top1_mass']:.1f}% "
            f"rho={row['tension_spearman_vs_unregularized']:.3f} "
            f"q95x={row['q95_residual_ratio']:.3f} "
            f"iters={row['optimizer_iterations']} {row['runtime_seconds']:.1f}s",
            flush=True
        )

        admiss=(row["q95_residual_ratio"]<=1.25 and row["tension_spearman_vs_unregularized"]>=0.85)
        if not admiss:
            bad+=1
        else:
            bad=0
        if bad>=2:
            print("        early stop: two consecutive inadmissible lambdas",flush=True)
            break

    df=pd.DataFrame(rows)
    admiss=df[(df.q95_residual_ratio<=1.25)&(df.tension_spearman_vs_unregularized>=0.85)].copy()
    if admiss.empty:
        return df,None,None,cache

    score=admiss.neff_fraction-admiss.top1_mass
    chosen=float(admiss.loc[score.idxmax(),"lambda_tau"])
    return df,chosen,sols[chosen],cache
