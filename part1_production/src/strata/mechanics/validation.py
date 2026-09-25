from __future__ import annotations
import numpy as np

def validate(edges, pressure, stress, sol):
    tau=edges["tension_like"].to_numpy(float) if len(edges) else np.array([])
    p=pressure["pressure_like"].to_numpy(float)
    r=pressure["force_balance_residual"].to_numpy(float)
    finite_tau=float(np.isfinite(tau).mean()) if len(tau) else 1.0
    finite_p=float(np.isfinite(p).mean()) if len(p) else 1.0
    finite_r=float(np.isfinite(r).mean()) if len(r) else 1.0
    stress_valid=float(stress["stress_valid"].mean()) if len(stress) else 0.0
    ok=(finite_tau==1.0 and finite_p==1.0 and finite_r==1.0 and np.isfinite(sol["condition_proxy"]) and stress_valid>0.99)
    return {
        "n_mechanics_edges":int(len(edges)),
        "n_mechanics_cells":int(len(pressure)),
        "finite_tension_fraction":finite_tau,
        "finite_pressure_fraction":finite_p,
        "finite_residual_fraction":finite_r,
        "median_force_balance_residual":float(np.nanmedian(r)) if len(r) else None,
        "q95_force_balance_residual":float(np.nanquantile(r,0.95)) if len(r) else None,
        "condition_proxy":float(sol["condition_proxy"]),
        "lsqr_istop":int(sol["lsqr_istop"]),
        "lsqr_iterations":int(sol["lsqr_iterations"]),
        "stress_valid_fraction":stress_valid,
        "status":"PASS" if ok else "FAIL",
    }
