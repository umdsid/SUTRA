from __future__ import annotations
import numpy as np


def certify(edges,junctions,pressure,stress,sol,reconstruction_fraction):
    tau=edges.tension_like.to_numpy(float)
    p=pressure.pressure_like.to_numpy(float)
    nr=np.asarray(sol["normalized_junction_residual"],dtype=float)

    finite = (
        np.isfinite(tau).all()
        and np.isfinite(p).all()
        and np.isfinite(nr).all()
    )

    mean_tau=float(np.mean(tau)) if len(tau) else 0.0
    std_tau=float(np.std(tau)) if len(tau) else 0.0
    std_p=float(np.std(p)) if len(p) else 0.0
    mech_norm=float(np.linalg.norm(np.concatenate([tau,p]))) if (len(tau)+len(p)) else 0.0

    # Nontriviality is a property of the inferred field, not optimizer iteration count.
    nontrivial = (
        len(tau)>0
        and abs(mean_tau-1.0) <= 0.05
        and mech_norm > 1e-6
        and std_tau > 1e-6
        and std_p > 1e-8
    )

    nonnegative = bool(np.all(tau >= -1e-10))
    stress_valid=float(stress.stress_valid.mean()) if len(stress) else 0.0
    q50=float(np.median(nr)) if len(nr) else float("inf")
    q95=float(np.quantile(nr,0.95)) if len(nr) else float("inf")

    geometry_ok = reconstruction_fraction >= 0.90
    residual_ok = q50 <= 0.50 and q95 <= 2.0
    solver_ok = bool(sol["success"])

    passed = (
        finite and nontrivial and nonnegative and geometry_ok
        and residual_ok and solver_ok and stress_valid>=0.99
    )

    return {
        "reconstruction_fraction":float(reconstruction_fraction),
        "n_mechanics_edges":int(len(edges)),
        "n_junctions":int(len(junctions)),
        "junction_degree_ge3_fraction":float((junctions.n_incident_interfaces>=3).mean()) if len(junctions) else 0.0,
        "mean_tension":mean_tau,
        "std_tension":std_tau,
        "mean_pressure":float(np.mean(p)) if len(p) else None,
        "std_pressure":std_p,
        "mechanical_solution_norm":mech_norm,
        "median_normalized_junction_residual":q50,
        "q95_normalized_junction_residual":q95,
        "stress_valid_fraction":stress_valid,
        "solver_success":solver_ok,
        "solver_iterations":int(sol["n_iterations"]),
        "finite_solution":bool(finite),
        "nontrivial_solution":bool(nontrivial),
        "nonnegative_tension":bool(nonnegative),
        "geometry_ok":bool(geometry_ok),
        "residual_ok":bool(residual_ok),
        "status":"PASS" if passed else "FAIL",
    }
