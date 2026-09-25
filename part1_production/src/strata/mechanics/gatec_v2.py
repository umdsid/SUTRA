from __future__ import annotations
import numpy as np


def paired_bootstrap_improvement(fit,baseline,n_boot=300,seed=17):
    fit=np.asarray(fit,float); base=np.asarray(baseline,float)
    d=base-fit
    finite=np.isfinite(d)
    d=d[finite]
    if len(d)==0:
        return {"median_improvement":-np.inf,"ci95_low":-np.inf,"ci95_high":-np.inf,"fraction_improved":0.0}
    rng=np.random.default_rng(seed)
    meds=np.empty(n_boot,float)
    N=len(d)
    for b in range(n_boot):
        idx=rng.integers(0,N,size=N)
        meds[b]=np.median(d[idx])
    return {
        "median_improvement":float(np.median(d)),
        "ci95_low":float(np.quantile(meds,0.025)),
        "ci95_high":float(np.quantile(meds,0.975)),
        "fraction_improved":float(np.mean(d>0)),
    }


def certify(edges,junctions,pressure,stress,sol,reconstruction_fraction):
    tau=np.asarray(sol["tension"],float)
    p=np.asarray(sol["pressure"],float)
    fit=np.asarray(sol["junction_residual"],float)
    base=np.asarray(sol["baseline_junction_residual"],float)

    finite=bool(np.isfinite(tau).all() and np.isfinite(p).all() and np.isfinite(fit).all() and np.isfinite(base).all())
    nonnegative=bool(np.all(tau>=-1e-10))
    exact_scale=bool(abs(float(np.mean(tau))-1.0)<=1e-8)
    exact_pressure_gauge=bool(abs(float(np.mean(p)))<=1e-8)
    nontrivial=bool(np.std(tau)>1e-6 and np.std(p)>1e-8 and np.linalg.norm(np.r_[tau,p])>1e-6)

    imp=paired_bootstrap_improvement(fit,base)
    q95_fit=float(np.quantile(fit,0.95)) if len(fit) else np.inf
    q95_base=float(np.quantile(base,0.95)) if len(base) else np.inf
    med_fit=float(np.median(fit)) if len(fit) else np.inf
    med_base=float(np.median(base)) if len(base) else np.inf

    # Data-relative residual gate: fitted mechanics must improve the same
    # measured junction geometry beyond the unit-tension / zero-pressure baseline.
    residual_ok=bool(
        imp["ci95_low"]>0.0
        and imp["fraction_improved"]>0.5
        and med_fit<med_base
        and q95_fit<=q95_base
    )

    geometry_ok=bool(reconstruction_fraction>=0.90)
    stress_valid=float(stress.stress_valid.mean()) if len(stress) else 0.0

    # scipy may hit max_iter despite a usable bounded solution; success is
    # accepted either by success flag or sufficiently small first-order optimality.
    optimizer_ok=bool(sol["success"] or sol["optimality"]<=1e-3)

    passed=all([
        finite,nonnegative,exact_scale,exact_pressure_gauge,nontrivial,
        residual_ok,geometry_ok,stress_valid>=0.99,optimizer_ok
    ])

    return {
        "reconstruction_fraction":float(reconstruction_fraction),
        "n_mechanics_edges":int(len(edges)),
        "n_junctions":int(len(junctions)),
        "junction_degree_ge3_fraction":float((junctions.n_incident_interfaces>=3).mean()) if len(junctions) else 0.0,
        "mean_tension":float(np.mean(tau)),
        "std_tension":float(np.std(tau)),
        "mean_pressure":float(np.mean(p)),
        "std_pressure":float(np.std(p)),
        "median_junction_residual":med_fit,
        "q95_junction_residual":q95_fit,
        "baseline_median_junction_residual":med_base,
        "baseline_q95_junction_residual":q95_base,
        **imp,
        "stress_valid_fraction":stress_valid,
        "optimizer_success":bool(sol["success"]),
        "optimizer_optimality":float(sol["optimality"]),
        "optimizer_iterations":int(sol["n_iterations"]),
        "finite_solution":finite,
        "nonnegative_tension":nonnegative,
        "exact_tension_scale":exact_scale,
        "exact_pressure_gauge":exact_pressure_gauge,
        "nontrivial_solution":nontrivial,
        "geometry_ok":geometry_ok,
        "residual_improvement_ok":residual_ok,
        "optimizer_ok":optimizer_ok,
        "status":"PASS" if passed else "FAIL",
    }
