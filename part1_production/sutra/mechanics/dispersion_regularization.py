from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from strata.mechanics.warmstart_regularization import build_cache


@dataclass(frozen=True)
class DispersionConfig:
    maxiter: int = 250
    gtol: float = 1e-5
    mu_scale: float = 100.0
    mu_gauge: float = 100.0
    ridge_p: float = 1e-8


def _objective_grad(x, cache, lam, weights, cfg: DispersionConfig):
    m, n = cache.m, cache.n
    tau = x[:m]
    p = x[m:]

    Ax = cache.A @ x
    val = 0.5 * float(np.dot(Ax, Ax))
    grad = cache.A.T @ Ax

    # Confidence-weighted dispersion around normalized mean tension 1.
    dt = tau - 1.0
    val += 0.5 * lam * float(np.dot(weights * dt, dt))
    grad[:m] += lam * weights * dt

    mt = float(np.mean(tau))
    mp = float(np.mean(p)) if n else 0.0

    ds = mt - 1.0
    val += 0.5 * cfg.mu_scale * ds * ds
    grad[:m] += cfg.mu_scale * ds / max(m, 1)

    if n:
        val += 0.5 * cfg.mu_gauge * mp * mp
        grad[m:] += cfg.mu_gauge * mp / max(n, 1)

        val += 0.5 * cfg.ridge_p * float(np.dot(p, p))
        grad[m:] += cfg.ridge_p * p

    return val, np.asarray(grad, float)


def solve_dispersion(cache, lam, base_tau, base_p, weights, cfg=DispersionConfig()):
    m, n = cache.m, cache.n
    x0 = np.r_[np.asarray(base_tau, float), np.asarray(base_p, float)]
    bounds = [(0.0, None)] * m + [(None, None)] * n

    t0 = time.time()
    res = minimize(
        fun=lambda x: _objective_grad(x, cache, lam, weights, cfg),
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": cfg.maxiter,
            "gtol": cfg.gtol,
            "ftol": 1e-12,
            "maxls": 30,
        },
    )

    tau = res.x[:m].copy()
    p = res.x[m:].copy()

    # Exact gauges after solve.
    mt = float(np.mean(tau))
    if mt <= 1e-12:
        raise RuntimeError("vanishing tension scale")
    tau /= mt
    p /= mt
    p -= float(np.mean(p))

    xx = np.r_[tau, p]
    force = cache.A @ xx
    jr = np.sqrt(force[0::2] ** 2 + force[1::2] ** 2)

    return {
        "tension": tau,
        "pressure": p,
        "junction_residual": jr,
        "success": bool(res.success),
        "message": str(res.message),
        "optimality": float(np.linalg.norm(res.jac, np.inf)),
        "n_iterations": int(res.nit),
        "runtime_seconds": float(time.time() - t0),
    }


def confidence_weights(edges: pd.DataFrame):
    """
    High-confidence geometry is allowed slightly more heterogeneity.
    Admissible geometry receives slightly stronger shrinkage toward tau=1.
    """
    rank = edges["confidence_class"].astype(str)
    w = np.ones(len(edges), dtype=float)
    w[rank == "high_confidence"] = 0.75
    w[rank == "admissible"] = 1.0
    return w


def neff_fraction(t):
    t = np.asarray(t, float)
    return float((np.sum(t) ** 2) / (np.sum(t * t) + 1e-15) / max(len(t), 1))


def top_mass(t, frac):
    t = np.asarray(t, float)
    n = len(t)
    if n == 0:
        return 0.0
    k = max(1, int(math.ceil(frac * n)))
    idx = np.argpartition(t, -k)[-k:]
    return float(np.sum(t[idx]) / (np.sum(t) + 1e-15))


def rho(a, b):
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else math.nan


def evaluate(lam, sol, base_resid):
    r = sol["junction_residual"]
    return {
        "lambda_tau": float(lam),
        "neff_fraction": neff_fraction(sol["tension"]),
        "top1_mass": top_mass(sol["tension"], 0.01),
        "top5_mass": top_mass(sol["tension"], 0.05),
        "median_residual_ratio": float(
            np.median(r) / (np.median(base_resid) + 1e-15)
        ),
        "q95_residual_ratio": float(
            np.quantile(r, 0.95) / (np.quantile(base_resid, 0.95) + 1e-15)
        ),
        "optimizer_success": bool(sol["success"]),
        "optimizer_iterations": int(sol["n_iterations"]),
        "runtime_seconds": float(sol["runtime_seconds"]),
    }


def calibrate_dispersion(edges, cell_ids, n_junctions, base_tau, base_p, base_resid):
    cache = build_cache(edges, cell_ids, n_junctions)
    weights = confidence_weights(edges)

    # Broad but cheap continuation sweep.
    lambdas = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    rows = []
    sols = {}
    previous_tau = None
    bad = 0

    for lam in lambdas:
        print(f"        dispersion solve lambda={lam:.1e} ...", flush=True)
        sol = solve_dispersion(cache, lam, base_tau, base_p, weights)
        row = evaluate(lam, sol, base_resid)

        if previous_tau is None:
            row["spearman_vs_previous_regularized"] = math.nan
        else:
            row["spearman_vs_previous_regularized"] = rho(previous_tau, sol["tension"])

        rows.append(row)
        sols[lam] = sol
        previous_tau = sol["tension"].copy()

        print(
            f"          done: Neff={100*row['neff_fraction']:.2f}% "
            f"top1={100*row['top1_mass']:.1f}% "
            f"rho_prev={row['spearman_vs_previous_regularized'] if np.isfinite(row['spearman_vs_previous_regularized']) else float('nan'):.3f} "
            f"q95x={row['q95_residual_ratio']:.3f} "
            f"iters={row['optimizer_iterations']} "
            f"{row['runtime_seconds']:.1f}s",
            flush=True,
        )

        if row["q95_residual_ratio"] > 2.0:
            bad += 1
        else:
            bad = 0

        if bad >= 2:
            print("        early stop: two consecutive q95 ratios > 2", flush=True)
            break

    sweep = pd.DataFrame(rows)

    # Plateau rule:
    # - residual q95 no more than 25% worse than Gate C;
    # - adjacent regularized fields stable with Spearman >=0.90;
    # - concentration must materially improve relative to first tested point.
    base_neff = neff_fraction(base_tau)
    base_top1 = top_mass(base_tau, 0.01)

    candidates = sweep.copy()
    candidates["improved_concentration"] = (
        (candidates["neff_fraction"] >= 3.0 * base_neff)
        & (candidates["top1_mass"] <= 0.80 * base_top1)
    )
    candidates["residual_ok"] = candidates["q95_residual_ratio"] <= 1.25
    candidates["plateau_stable"] = (
        candidates["spearman_vs_previous_regularized"].fillna(0.0) >= 0.90
    )

    good = candidates[
        candidates["improved_concentration"]
        & candidates["residual_ok"]
        & candidates["plateau_stable"]
    ].copy()

    if good.empty:
        return sweep, None, None, cache

    # Choose weakest lambda satisfying all conditions.
    chosen = float(good.sort_values("lambda_tau").iloc[0]["lambda_tau"])
    return sweep, chosen, sols[chosen], cache
