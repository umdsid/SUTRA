from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sutra.mechanics.regularized_solver import solve_regularized,RegularizedConfig


def _neff(t):
    t=np.asarray(t,float)
    return float(np.sum(t)**2/(np.sum(t*t)+1e-15))

def _topmass(t,frac):
    t=np.asarray(t,float); n=len(t); k=max(1,int(math.ceil(frac*n)))
    idx=np.argpartition(t,-k)[-k:]
    return float(np.sum(t[idx])/(np.sum(t)+1e-15))

def _rho(a,b):
    r=spearmanr(a,b).statistic
    return float(r) if np.isfinite(r) else math.nan

def calibrate_lambda(edges,cell_ids,n_junctions,baseline_tau,baseline_resid):
    lambdas=[0.0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1]
    rows=[]; sols={}
    bq95=float(np.quantile(baseline_resid,0.95))
    bmed=float(np.median(baseline_resid))
    N=len(edges)

    for lam in lambdas:
        sol=solve_regularized(edges,cell_ids,n_junctions,RegularizedConfig(lambda_tau=lam))
        t=sol["tension"]; r=sol["junction_residual"]
        row={
            "lambda_tau":lam,
            "neff_fraction":_neff(t)/max(N,1),
            "top1_mass":_topmass(t,0.01),
            "top5_mass":_topmass(t,0.05),
            "tension_spearman_vs_unregularized":_rho(baseline_tau,t),
            "median_residual":float(np.median(r)),
            "q95_residual":float(np.quantile(r,0.95)),
            "median_residual_ratio":float(np.median(r)/(bmed+1e-15)),
            "q95_residual_ratio":float(np.quantile(r,0.95)/(bq95+1e-15)),
            "optimizer_success":sol["success"],
            "optimizer_optimality":sol["optimality"],
            "optimizer_iterations":sol["n_iterations"],
        }
        rows.append(row); sols[lam]=sol

    sweep=pd.DataFrame(rows)

    # Data-driven selection: among solutions whose residual q95 degrades by no
    # more than 25% and rank correlation remains >=0.85, choose the earliest
    # lambda at which concentration improvement plateaus.
    admiss=sweep[
        (sweep.q95_residual_ratio<=1.25) &
        (sweep.tension_spearman_vs_unregularized>=0.85)
    ].copy()

    if admiss.empty:
        chosen=None
    else:
        # gain in effective support relative to previous admissible point
        admiss["neff_gain"]=admiss.neff_fraction.diff().fillna(np.inf)
        chosen=None
        vals=admiss.reset_index(drop=True)
        for k in range(2,len(vals)):
            g1=vals.loc[k-1,"neff_gain"]; g2=vals.loc[k,"neff_gain"]
            if np.isfinite(g1) and np.isfinite(g2) and g1<0.01 and g2<0.01:
                chosen=float(vals.loc[k-1,"lambda_tau"]); break
        if chosen is None:
            score=vals.neff_fraction - vals.top1_mass
            chosen=float(vals.loc[score.idxmax(),"lambda_tau"])

    return sweep,chosen,sols.get(chosen) if chosen is not None else None
