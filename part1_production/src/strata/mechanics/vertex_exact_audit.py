from __future__ import annotations
import math
import numpy as np


def neff_fraction(t):
    t=np.asarray(t,float)
    return float(np.sum(t)**2/(np.sum(t*t)+1e-15)/max(len(t),1))


def top_mass(t,frac=0.01):
    t=np.asarray(t,float)
    n=len(t)
    if n==0:return 0.0
    k=max(1,int(math.ceil(frac*n)))
    idx=np.argpartition(t,-k)[-k:]
    return float(np.sum(t[idx])/(np.sum(t)+1e-15))


def summarize_vertex_mechanics(
    sol,
    geometry_diag,
    n_total_cells,
    n_domain_cells,
    n_total_edges,
    n_domain_edges,
):
    tau=np.asarray(sol["tension"],float)
    p=np.asarray(sol["pressure"],float)
    r=np.asarray(sol["vertex_residual"],float)

    geom_valid=float(geometry_diag["geometry_valid"].mean()) if len(geometry_diag) else 0.0
    domain_cell_fraction=n_domain_cells/max(n_total_cells,1)
    domain_edge_fraction=n_domain_edges/max(n_total_edges,1)

    return {
        "solver_success":bool(sol["success"]),
        "solver_iterations":int(sol["iterations"]),
        "solver_optimality":float(sol["optimality"]),
        "runtime_seconds":float(sol["runtime_seconds"]),
        "finite_solution":bool(np.isfinite(tau).all() and np.isfinite(p).all() and np.isfinite(r).all()),
        "nonnegative_tension":bool(np.all(tau>=-1e-10)),
        "mean_tension":float(np.mean(tau)) if len(tau) else None,
        "std_tension":float(np.std(tau)) if len(tau) else None,
        "mean_pressure":float(np.mean(p)) if len(p) else None,
        "std_pressure":float(np.std(p)) if len(p) else None,
        "median_vertex_residual":float(np.median(r)) if len(r) else None,
        "q95_vertex_residual":float(np.quantile(r,0.95)) if len(r) else None,
        "tension_neff_fraction":neff_fraction(tau),
        "tension_top1_mass":top_mass(tau,0.01),
        "geometry_valid_cell_fraction":geom_valid,
        "mechanics_domain_cell_fraction":float(domain_cell_fraction),
        "mechanics_domain_edge_fraction":float(domain_edge_fraction),
    }
