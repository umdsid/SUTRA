#!/usr/bin/env python3

from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd
import ot


ROOT = Path.home() / "Desktop" / "SUTRA"
V2 = ROOT / "results" / "Fig5_Disease_Trajectories" / "gw_geodesic_dense_v5"
OUT = ROOT / "results" / "Fig5_Disease_Trajectories" / "gw_multistart_dense_v5"

K = 512
N_RANDOM_STARTS = 12
SEED = 20260920
EPS = 1e-12

# We are auditing the already materialized V2 sketches only.
# No hierarchy, quotient graph, FPS, or shortest paths are recomputed.


def manual_square_loss_gw_objective(C1, C2, p, q, T):
    """
    Square-loss GW objective:
        sum_ijkl (C1_ik - C2_jl)^2 T_ij T_kl
    evaluated without constructing the four-index tensor.
    """
    a = np.sum((C1 * C1) * np.outer(p, p))
    b = np.sum((C2 * C2) * np.outer(q, q))
    cross = np.sum(T * (C1 @ T @ C2.T))
    val = float(a + b - 2.0 * cross)

    if val < 0 and abs(val) < 1e-10:
        val = 0.0
    return val


def infer_u_from_name(path):
    """
    Fallback parser only. Prefer u stored inside NPZ.
    """
    s = path.stem.lower()

    pats = [
        r"u[_-]?([0-9]+(?:\.[0-9]+)?)",
        r"u([0-9]+)p([0-9]+)",
    ]

    m = re.search(pats[0], s)
    if m:
        x = float(m.group(1))
        if x > 1:
            x /= 100.0
        return x

    m = re.search(pats[1], s)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")

    return np.nan


def scalar_from_npz(z, keys, default=np.nan):
    for k in keys:
        if k in z.files:
            x = np.asarray(z[k])
            if x.size == 1:
                return float(x.reshape(-1)[0])
    return default


def load_v2_index():
    """
    Bind the frozen K=512 NPZ sketches to the authoritative V2 trajectory
    table.  The NPZ filenames encode target u and K; the CSV contains the
    exact attained reference/disease hierarchy coordinates.
    """
    traj_path = V2 / "gw_scale_trajectory.csv"
    if not traj_path.exists():
        raise RuntimeError(f"Missing V2 trajectory table: {traj_path}")

    traj = pd.read_csv(traj_path)

    required = {
        "u_target",
        "u_reference",
        "u_disease",
        "K",
    }
    missing = required - set(traj.columns)
    if missing:
        raise RuntimeError(
            f"{traj_path.name} missing columns: {sorted(missing)}"
        )

    traj = traj[traj["K"].astype(int) == K].copy()
    traj = traj.sort_values("u_target").reset_index(drop=True)

    if len(traj) != 25:
        raise RuntimeError(
            f"Expected 25 dense K={K} rows in {traj_path.name}; found {len(traj)}"
        )

    rows = []

    for _, r in traj.iterrows():
        u = float(r["u_target"])

        # Production V2 naming convention:
        # 0.08 -> u0p08, etc.
        utag = f"{u:.2f}".replace(".", "p")
        pth = V2 / f"gw_state_u{utag}_K{K}.npz"

        if not pth.exists():
            raise RuntimeError(f"Missing frozen V2 sketch: {pth}")

        with np.load(pth, allow_pickle=False) as z:
            needed = {
                "C_reference",
                "C_disease",
                "p_reference",
                "p_disease",
                "anchors_reference",
                "anchors_disease",
                "owner_reference",
                "owner_disease",
                "dominant_object_labels_reference",
                "dominant_object_labels_disease",
                "dominant_original_indices_reference",
                "dominant_original_indices_disease",
                "coverage_reference",
                "coverage_disease",
            }

            miss = needed - set(z.files)
            if miss:
                raise RuntimeError(
                    f"{pth.name} missing arrays: {sorted(miss)}"
                )

            if z["C_reference"].shape != (K, K):
                raise RuntimeError(
                    f"{pth.name}: reference metric has shape "
                    f"{z['C_reference'].shape}"
                )

            if z["C_disease"].shape != (K, K):
                raise RuntimeError(
                    f"{pth.name}: disease metric has shape "
                    f"{z['C_disease'].shape}"
                )

        rows.append({
            "path": str(pth),
            "K": K,
            "u_target": u,
            "u_reference": float(r["u_reference"]),
            "u_disease": float(r["u_disease"]),
        })

    return pd.DataFrame(rows)


def validate_metric_measure(C, p, name):
    C = np.asarray(C, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64).reshape(-1)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"{name}: C not square: {C.shape}")

    if len(p) != C.shape[0]:
        raise ValueError(
            f"{name}: mass length {len(p)} != metric size {C.shape[0]}"
        )

    if not np.all(np.isfinite(C)):
        raise ValueError(f"{name}: nonfinite metric")

    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name}: nonfinite mass")

    if np.min(p) < -1e-14:
        raise ValueError(f"{name}: negative mass")

    if not np.allclose(C, C.T, atol=1e-10, rtol=0):
        raise ValueError(f"{name}: asymmetric metric")

    if not np.allclose(np.diag(C), 0.0, atol=1e-10, rtol=0):
        raise ValueError(f"{name}: nonzero metric diagonal")

    p = np.maximum(p, 0.0)
    p /= p.sum()

    return C, p


def independent_coupling(p, q):
    return np.outer(p, q)


def random_feasible_coupling(p, q, rng):
    """
    Positive random matrix projected approximately onto the transport
    polytope by iterative proportional fitting.

    This provides genuinely different feasible GW initializations while
    preserving the exact marginals.
    """
    n = len(p)
    m = len(q)

    # Lognormal gives useful heterogeneity without extreme underflow.
    X = np.exp(rng.normal(0.0, 1.0, size=(n, m)))
    X += 1e-15

    for _ in range(1000):
        rs = X.sum(axis=1)
        X *= (p / np.maximum(rs, EPS))[:, None]

        cs = X.sum(axis=0)
        X *= (q / np.maximum(cs, EPS))[None, :]

        if _ % 25 == 0:
            er = np.max(np.abs(X.sum(axis=1) - p))
            ec = np.max(np.abs(X.sum(axis=0) - q))
            if max(er, ec) < 1e-12:
                break

    # Final balancing passes.
    for _ in range(10):
        X *= (p / np.maximum(X.sum(axis=1), EPS))[:, None]
        X *= (q / np.maximum(X.sum(axis=0), EPS))[None, :]

    er = float(np.max(np.abs(X.sum(axis=1) - p)))
    ec = float(np.max(np.abs(X.sum(axis=0) - q)))

    if max(er, ec) > 1e-9:
        raise RuntimeError(
            f"Could not construct feasible random coupling: "
            f"rowerr={er}, colerr={ec}"
        )

    return X


def run_one(C1, C2, p, q, G0):
    t0 = time.time()

    T, log = ot.gromov.gromov_wasserstein(
        C1,
        C2,
        p,
        q,
        loss_fun="square_loss",
        G0=G0,
        log=True,
        armijo=False,
    )

    elapsed = time.time() - t0

    T = np.asarray(T, dtype=np.float64)

    row_err = float(np.max(np.abs(T.sum(axis=1) - p)))
    col_err = float(np.max(np.abs(T.sum(axis=0) - q)))
    coupling_mass = float(T.sum())

    obj_manual = manual_square_loss_gw_objective(
        C1, C2, p, q, T
    )

    obj_pot = np.nan
    if isinstance(log, dict):
        for key in ("gw_dist", "loss"):
            if key in log:
                x = log[key]
                if np.ndim(x) == 0:
                    obj_pot = float(x)
                elif len(x):
                    obj_pot = float(x[-1])
                break

    return {
        "T": T,
        "objective": obj_manual,
        "gw_distance": float(np.sqrt(max(obj_manual, 0.0))),
        "pot_objective": obj_pot,
        "objective_delta": (
            abs(obj_manual - obj_pot)
            if np.isfinite(obj_pot)
            else np.nan
        ),
        "coupling_mass": coupling_mass,
        "row_marginal_error": row_err,
        "column_marginal_error": col_err,
        "elapsed_seconds": elapsed,
    }


def conditional_disease_distortion(C1, C2, p, q, T):
    """
    Disease-side conditional expected square-loss GW distortion.

    For disease landmark j:

      delta_j =
        (1 / q_j)
        sum_i T_ij
        sum_kl T_kl (C1_ik - C2_jl)^2

    evaluated algebraically without a four-index tensor.

    Weighted average over disease masses recovers the global GW^2
    objective, providing an additional exact audit.
    """
    C1sq_p = (C1 * C1) @ p
    C2sq_q = (C2 * C2) @ q

    # cross[i,j] = sum_kl C1[i,k] T[k,l] C2[j,l]
    cross = C1 @ T @ C2.T

    local_ij = (
        C1sq_p[:, None]
        + C2sq_q[None, :]
        - 2.0 * cross
    )

    qhat = T.sum(axis=0)

    numer = np.sum(T * local_ij, axis=0)

    out = np.full(len(qhat), np.nan, dtype=np.float64)
    good = qhat > EPS
    out[good] = numer[good] / qhat[good]

    tiny = (out < 0) & (out > -1e-10)
    out[tiny] = 0.0

    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("SUTRA FIGURE 5 — GW MULTISTART OPTIMIZATION AUDIT V3")
    print("=" * 100)
    print("Frozen input:", V2)
    print("Output:", OUT)
    print("POT:", ot.__version__)
    print("K:", K)
    print("Random starts per scale:", N_RANDOM_STARTS)

    index = load_v2_index()
    index.to_csv(OUT / "v2_input_index.csv", index=False)

    print("\nInput index")
    print(index.to_string(index=False))

    rng_master = np.random.default_rng(SEED)

    all_rows = []
    best_rows = []

    for _, rr in index.iterrows():
        pth = Path(rr["path"])
        u = float(rr["u_target"])

        print("\n" + "-" * 100)
        print(f"u = {u:.2f}")
        print(pth.name)
        print("-" * 100)

        with np.load(pth, allow_pickle=False) as z:
            C1, p = validate_metric_measure(
                z["C_reference"],
                z["p_reference"],
                "reference",
            )
            C2, q = validate_metric_measure(
                z["C_disease"],
                z["p_disease"],
                "disease",
            )

        starts = []

        # Start 0 = canonical independent coupling.
        starts.append(("independent", independent_coupling(p, q)))

        # Remaining starts = randomized feasible couplings.
        for s in range(N_RANDOM_STARTS):
            seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            starts.append(
                (
                    f"random_{s+1:02d}",
                    random_feasible_coupling(p, q, rng),
                )
            )

        solutions = []

        for start_idx, (start_name, G0) in enumerate(starts):
            print(
                f"  {start_idx+1:02d}/{len(starts):02d} "
                f"{start_name:>12s}",
                end="",
                flush=True,
            )

            res = run_one(C1, C2, p, q, G0)

            print(
                f"  GW={res['gw_distance']:.8f}"
                f"  GW2={res['objective']:.10f}"
                f"  mass={res['coupling_mass']:.12f}"
                f"  dt={res['elapsed_seconds']:.2f}s"
            )

            if res["row_marginal_error"] > 1e-8:
                raise RuntimeError(
                    f"row marginal failure at u={u}, {start_name}"
                )

            if res["column_marginal_error"] > 1e-8:
                raise RuntimeError(
                    f"column marginal failure at u={u}, {start_name}"
                )

            row = {
                "u_target": u,
                "u_reference": float(rr["u_reference"]),
                "u_disease": float(rr["u_disease"]),
                "K": K,
                "start_index": start_idx,
                "start_name": start_name,
                "objective": res["objective"],
                "gw_distance": res["gw_distance"],
                "pot_objective": res["pot_objective"],
                "objective_delta": res["objective_delta"],
                "coupling_mass": res["coupling_mass"],
                "row_marginal_error": res["row_marginal_error"],
                "column_marginal_error": res["column_marginal_error"],
                "elapsed_seconds": res["elapsed_seconds"],
            }

            all_rows.append(row)
            solutions.append((row, res["T"]))

        solutions.sort(key=lambda x: x[0]["objective"])
        best_row, best_T = solutions[0]

        objs = np.array(
            [x[0]["objective"] for x in solutions],
            dtype=float,
        )
        dists = np.sqrt(np.maximum(objs, 0.0))

        independent_obj = next(
            x[0]["objective"]
            for x in solutions
            if x[0]["start_name"] == "independent"
        )

        independent_dist = np.sqrt(max(independent_obj, 0.0))

        summary = {
            "u_target": u,
            "u_reference": float(rr["u_reference"]),
            "u_disease": float(rr["u_disease"]),
            "K": K,
            "n_starts": len(solutions),
            "best_start": best_row["start_name"],
            "best_objective": float(objs.min()),
            "best_gw_distance": float(dists.min()),
            "median_objective": float(np.median(objs)),
            "median_gw_distance": float(np.median(dists)),
            "worst_objective": float(objs.max()),
            "worst_gw_distance": float(dists.max()),
            "objective_range": float(objs.max() - objs.min()),
            "gw_distance_range": float(dists.max() - dists.min()),
            "relative_gw_range_vs_best": float(
                (dists.max() - dists.min())
                / max(dists.min(), EPS)
            ),
            "independent_objective": float(independent_obj),
            "independent_gw_distance": float(independent_dist),
            "independent_minus_best_objective": float(
                independent_obj - objs.min()
            ),
            "independent_minus_best_gw": float(
                independent_dist - dists.min()
            ),
        }

        best_rows.append(summary)

        print(
            "  BEST:",
            summary["best_start"],
            "GW =",
            summary["best_gw_distance"],
            "range =",
            summary["gw_distance_range"],
            "relative range =",
            summary["relative_gw_range_vs_best"],
        )

        # Save best coupling and disease conditional distortion for V4.
        distortion = conditional_disease_distortion(
            C1, C2, p, q, best_T
        )

        # Exact decomposition audit:
        # disease-mass-weighted local distortion must recover global GW^2.
        local_reconstruction = float(
            np.sum(q * distortion)
        )
        local_reconstruction_delta = abs(
            local_reconstruction - summary["best_objective"]
        )

        if local_reconstruction_delta > 1e-9:
            raise RuntimeError(
                f"Local distortion decomposition failed at u={u}: "
                f"{local_reconstruction_delta}"
            )

        print(
            "  local-distortion reconstruction delta =",
            local_reconstruction_delta,
        )

        with np.load(pth, allow_pickle=False) as zloc:
            np.savez_compressed(
                OUT / f"best_multistart_u_{u:.2f}_K{K}.npz",
                u_target=np.array(u),
                u_reference=np.array(float(rr["u_reference"])),
                u_disease=np.array(float(rr["u_disease"])),
                K=np.array(K),
                C_reference=C1,
                C_disease=C2,
                p_reference=p,
                p_disease=q,
                T_best=best_T,
                disease_conditional_distortion=distortion,
                best_objective=np.array(summary["best_objective"]),
                best_gw_distance=np.array(summary["best_gw_distance"]),
                local_distortion_reconstruction_delta=np.array(
                    local_reconstruction_delta
                ),
                anchors_reference=zloc["anchors_reference"],
                anchors_disease=zloc["anchors_disease"],
                owner_reference=zloc["owner_reference"],
                owner_disease=zloc["owner_disease"],
                dominant_object_labels_reference=zloc[
                    "dominant_object_labels_reference"
                ],
                dominant_object_labels_disease=zloc[
                    "dominant_object_labels_disease"
                ],
                dominant_original_indices_reference=zloc[
                    "dominant_original_indices_reference"
                ],
                dominant_original_indices_disease=zloc[
                    "dominant_original_indices_disease"
                ],
                coverage_reference=zloc["coverage_reference"],
                coverage_disease=zloc["coverage_disease"],
            )

    all_df = pd.DataFrame(all_rows)
    best_df = pd.DataFrame(best_rows).sort_values("u_target")

    all_df.to_csv(
        OUT / "multistart_all_runs.csv",
        index=False,
    )

    best_df.to_csv(
        OUT / "multistart_scale_summary.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("MULTISTART SCALE SUMMARY")
    print("=" * 100)

    cols = [
        "u_target",
        "best_start",
        "best_gw_distance",
        "median_gw_distance",
        "worst_gw_distance",
        "relative_gw_range_vs_best",
        "independent_gw_distance",
        "independent_minus_best_gw",
    ]

    print(best_df[cols].to_string(index=False))

    # Numerical audit criteria only; these do NOT constitute biological
    # significance tests.
    max_mass_err = max(
        abs(all_df["coupling_mass"] - 1.0).max(),
        all_df["row_marginal_error"].max(),
        all_df["column_marginal_error"].max(),
    )

    provenance = {
        "version": "fig5_gw_multistart_dense_v5",
        "input": str(V2),
        "K": K,
        "n_random_starts": N_RANDOM_STARTS,
        "plus_independent_start": True,
        "seed": SEED,
        "loss": "square_loss",
        "purpose": (
            "Optimization-stability audit of frozen SUTRA GW V2 "
            "metric-measure sketches. No hierarchy reconstruction."
        ),
        "max_transport_constraint_error": float(max_mass_err),
    }

    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )

    print("\nCOMPLETE")
    print(OUT)


if __name__ == "__main__":
    main()
