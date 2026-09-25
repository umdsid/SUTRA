from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import structural_rank
from scipy.stats import spearmanr
from sutra.mechanics.junction_calibration import cluster_for_tolerance
from sutra.mechanics.vertex_solver_v2 import SolverConfig, build_vertex_system, solve

def _q(x, probs=(0,0.01,0.05,0.25,0.5,0.75,0.95,0.99,1.0)):
    x=np.asarray(x,float)
    return {f"q{int(round(100*p)):02d}":float(np.quantile(x,p)) for p in probs}

def effective_support_nonnegative(x):
    x=np.asarray(x,float); den=float(np.sum(x*x))
    return 0.0 if den<=0 else float(np.sum(x)**2/den)

def gini_nonnegative(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)==0 or np.sum(x)<=0:return 0.0
    x=np.sort(np.maximum(x,0.0)); n=len(x)
    return float((2*np.sum(np.arange(1,n+1)*x)/(n*np.sum(x)))-((n+1)/n))

def baseline_field_summary(edges, pressure):
    tau=edges["tension_like"].to_numpy(float); p=pressure["pressure_like"].to_numpy(float); n=len(tau)
    neff=effective_support_nonnegative(tau); idx=np.argsort(tau)[::-1] if n else np.array([],dtype=int)
    k1=max(1,int(math.ceil(0.01*n))) if n else 0; k5=max(1,int(math.ceil(0.05*n))) if n else 0
    total=float(np.sum(tau))
    return {
        "tension_quantiles":_q(tau),"pressure_quantiles":_q(p),
        "tension_near_zero_fraction_le_1e-3":float(np.mean(tau<=1e-3)) if n else 1.0,
        "tension_cv":float(np.std(tau)/(np.mean(tau)+1e-15)) if n else None,
        "tension_gini":gini_nonnegative(tau),
        "tension_effective_support":neff,
        "tension_effective_support_fraction":float(neff/n) if n else 0.0,
        "tension_mass_top1pct":float(np.sum(tau[idx[:k1]])/total) if total>0 else 0.0,
        "tension_mass_top5pct":float(np.sum(tau[idx[:k5]])/total) if total>0 else 0.0,
        "pressure_sd":float(np.std(p)) if len(p) else 0.0,
    }

def structural_identifiability(edges, cell_ids, n_junctions):
    A=build_vertex_system(edges,cell_ids,n_junctions); sr=int(structural_rank(A)); nu=int(A.shape[1]); neq=int(A.shape[0])
    return {"n_equations":neq,"n_unknowns":nu,"structural_rank":sr,
            "structural_nullity_upper_bound":max(0,nu-sr),
            "structural_rank_fraction_of_unknowns":float(sr/nu) if nu else 1.0,
            "equation_to_unknown_ratio":float(neq/nu) if nu else None}

def _pearson(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)!=len(y) or len(x)<2 or np.std(x)<=0 or np.std(y)<=0:return math.nan
    return float(np.corrcoef(x,y)[0,1])

def _spearman(x,y):
    if len(x)!=len(y) or len(x)<2:return math.nan
    r=spearmanr(x,y).statistic
    return float(r) if np.isfinite(r) else math.nan

def top_overlap(x,y,fraction=0.05):
    x=np.asarray(x,float); y=np.asarray(y,float); n=len(x)
    if n==0 or len(y)!=n:return math.nan
    k=max(1,int(math.ceil(fraction*n)))
    a=set(np.argpartition(x,-k)[-k:].tolist()); b=set(np.argpartition(y,-k)[-k:].tolist())
    return float(len(a&b)/k)

def compare_fields(bt,bp,t,p):
    return {"tension_pearson":_pearson(bt,t),"tension_spearman":_spearman(bt,t),
            "pressure_pearson":_pearson(bp,p),"pressure_spearman":_spearman(bp,p),
            "top5_tension_overlap":top_overlap(bt,t,0.05),"top1_tension_overlap":top_overlap(bt,t,0.01),
            "relative_tension_l2":float(np.linalg.norm(t-bt)/(np.linalg.norm(bt)+1e-15)),
            "relative_pressure_l2":float(np.linalg.norm(p-bp)/(np.linalg.norm(bp)+1e-15))}

def perturbation_suite(raw_geom,base_geom,base_junctions,cell_ids,base_tau,base_p,chosen_jtol):
    rec=[]
    for mult in (0.25,0.5,2.0,4.0):
        cfg=SolverConfig(ridge_tension=1e-6*mult,ridge_pressure=1e-6*mult,max_iter=1000,tol=1e-5)
        s=solve(base_geom,cell_ids,len(base_junctions),cfg); c=compare_fields(base_tau,base_p,s["tension"],s["pressure"])
        rec.append({"perturbation_type":"ridge_multiplier","perturbation_value":mult,"n_junctions":len(base_junctions),"optimizer_success":bool(s["success"]),"optimizer_optimality":float(s["optimality"]),**c})
    for mult in (0.8,0.9):
        tol=float(chosen_jtol*mult); geom,junc,_=cluster_for_tolerance(raw_geom,tol)
        s=solve(geom,cell_ids,len(junc),SolverConfig(max_iter=1000,tol=1e-5)); c=compare_fields(base_tau,base_p,s["tension"],s["pressure"])
        rec.append({"perturbation_type":"junction_tolerance_multiplier","perturbation_value":mult,"junction_tolerance":tol,"n_junctions":len(junc),"optimizer_success":bool(s["success"]),"optimizer_optimality":float(s["optimality"]),**c})
    return pd.DataFrame(rec)

def certify_identifiability(summary,structural,pert):
    a={"minimum_tension_spearman":float(np.nanmin(pert.tension_spearman)),"minimum_pressure_spearman":float(np.nanmin(pert.pressure_spearman)),"minimum_top5_tension_overlap":float(np.nanmin(pert.top5_tension_overlap)),"maximum_relative_tension_l2":float(np.nanmax(pert.relative_tension_l2)),"maximum_relative_pressure_l2":float(np.nanmax(pert.relative_pressure_l2))}
    support=(summary["tension_effective_support_fraction"]>=0.01 and summary["tension_mass_top1pct"]<=0.90)
    stability=(a["minimum_tension_spearman"]>=0.85 and a["minimum_pressure_spearman"]>=0.80 and a["minimum_top5_tension_overlap"]>=0.70 and a["maximum_relative_tension_l2"]<=0.75 and a["maximum_relative_pressure_l2"]<=1.00)
    structural_ok=structural["structural_rank_fraction_of_unknowns"]>=0.50
    a.update({"support_ok":bool(support),"stability_ok":bool(stability),"structural_ok":bool(structural_ok),"status":"PASS" if support and stability and structural_ok else "FAIL"})
    return a
