from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import numpy as np
import pandas as pd
from scipy.sparse.linalg import lsqr, lsmr


@dataclass(frozen=True)
class ProductionSolveConfig:
    atol: float = 1e-12
    btol: float = 1e-12
    solver_agreement_tol: float = 1e-7
    maxiter_factor: int = 12


def solve_patch_pair(A, b, cfg: ProductionSolveConfig):
    """
    Solve the same least-squares problem with LSQR and LSMR.

    Neither solution is interpreted directly.  Observable linear quantities
    are invariant across the least-squares solution family; agreement of two
    independent Krylov solvers is retained as a numerical implementation check.
    """
    nvar = int(A.shape[1])
    maxiter = max(100, cfg.maxiter_factor * max(nvar, 1))

    q1 = lsqr(
        A, b,
        atol=cfg.atol, btol=cfg.btol,
        iter_lim=maxiter, show=False,
    )
    x1 = np.asarray(q1[0], dtype=float)

    q2 = lsmr(
        A, b,
        atol=cfg.atol, btol=cfg.btol,
        maxiter=maxiter, show=False,
    )
    x2 = np.asarray(q2[0], dtype=float)

    bnorm = max(float(np.linalg.norm(b)), 1.0)
    r1 = float(np.linalg.norm(A @ x1 - b) / bnorm)
    r2 = float(np.linalg.norm(A @ x2 - b) / bnorm)

    return x1, x2, {
        "lsqr_istop": int(q1[1]),
        "lsqr_iterations": int(q1[2]),
        "lsqr_relative_residual": r1,
        "lsqr_acond": float(q1[6]),
        "lsmr_istop": int(q2[1]),
        "lsmr_iterations": int(q2[2]),
        "lsmr_relative_residual": r2,
        "lsmr_conda": float(q2[6]),
    }


def relative_solver_disagreement(a: float, b: float) -> float:
    return float(abs(float(a) - float(b)) / max(1.0, abs(float(a)), abs(float(b))))


def interface_owner_map(cores: Iterable[Iterable[int]], interfaces: pd.DataFrame):
    """
    Deterministic core ownership.

    Every cell belongs to one partition core.  A cell-cell interface is owned
    by the core containing the smaller cell ID.  A cell-background interface
    is owned by the core containing its tissue cell.

    The rule is data-independent and does not select an owner based on whether
    a favorable observability result happens to occur in another patch.
    """
    cell_owner: Dict[int, int] = {}
    for pid, cells in enumerate(cores):
        for c in cells:
            c = int(c)
            if c in cell_owner:
                raise ValueError(f"cell {c} belongs to multiple cores")
            cell_owner[c] = int(pid)

    rows = []
    for r in interfaces.itertuples():
        eid = int(r.interface_id)
        if r.kind == "cell_cell":
            a, b = int(r.cell_i), int(r.cell_j)
            anchor = min(a, b)
        else:
            # Native geometry stores the tissue cell as cell_i for a
            # cell-background interface.
            anchor = int(r.cell_i)
        if anchor not in cell_owner:
            continue
        rows.append({
            "interface_id": eid,
            "owner_patch_id": int(cell_owner[anchor]),
            "owner_anchor_cell": int(anchor),
        })
    return pd.DataFrame(rows), cell_owner


def extract_patch_observables(
    patch_id: int,
    x1: np.ndarray,
    x2: np.ndarray,
    meta: dict,
    variable_mask: pd.DataFrame,
    contrast_mask: pd.DataFrame,
    cfg: ProductionSolveConfig,
):
    """
    Extract values only where v0.6.8 certified OBSERVABLE.

    Individual cell/boundary pressures are intentionally not exported as
    production values.  Tensions and cell-cell pressure contrasts are.
    """
    trows = []
    q = variable_mask[
        (variable_mask.patch_id.astype(int) == int(patch_id)) &
        (variable_mask.variable_class == "tension") &
        (variable_mask.rowspace_status == "OBSERVABLE")
    ]
    for r in q.itertuples():
        eid = int(r.object_id)
        if eid not in meta["eidx"]:
            continue
        k = int(meta["eidx"][eid])
        v1, v2 = float(x1[k]), float(x2[k])
        trows.append({
            "patch_id": int(patch_id),
            "interface_id": eid,
            "tension": 0.5 * (v1 + v2),
            "tension_lsqr": v1,
            "tension_lsmr": v2,
            "solver_disagreement": relative_solver_disagreement(v1, v2),
            "rowspace_residual_ratio": float(r.rowspace_residual_ratio),
            "status": "OBSERVABLE",
        })

    drows = []
    q = contrast_mask[
        (contrast_mask.patch_id.astype(int) == int(patch_id)) &
        (contrast_mask.rowspace_status == "OBSERVABLE")
    ]
    for r in q.itertuples():
        a, b = int(r.cell_i), int(r.cell_j)
        if a not in meta["cidx"] or b not in meta["cidx"]:
            continue
        ia, ib = int(meta["cidx"][a]), int(meta["cidx"][b])
        v1 = float(x1[ia] - x1[ib])
        v2 = float(x2[ia] - x2[ib])
        drows.append({
            "patch_id": int(patch_id),
            "interface_id": int(r.interface_id),
            "cell_i": a,
            "cell_j": b,
            "delta_p": 0.5 * (v1 + v2),
            "delta_p_lsqr": v1,
            "delta_p_lsmr": v2,
            "solver_disagreement": relative_solver_disagreement(v1, v2),
            "rowspace_residual_ratio": float(r.rowspace_residual_ratio),
            "status": "OBSERVABLE",
        })

    return pd.DataFrame(trows), pd.DataFrame(drows)


def core_owned_table(all_values: pd.DataFrame, owners: pd.DataFrame, value_kind: str):
    """
    Retain the observable estimate from the deterministic owner patch.
    Every physical interface appears at most once.
    """
    if len(owners) == 0:
        return pd.DataFrame()
    if len(all_values) == 0:
        out = owners.copy()
        out["production_status"] = "UNRESOLVED"
        return out

    merged = owners.merge(
        all_values,
        left_on=["interface_id", "owner_patch_id"],
        right_on=["interface_id", "patch_id"],
        how="left",
    )
    merged["production_status"] = np.where(
        merged["status"].eq("OBSERVABLE"),
        "OBSERVABLE",
        "UNRESOLVED",
    )
    if "patch_id" in merged:
        merged = merged.drop(columns=["patch_id"])
    return merged


def overlap_validation(values: pd.DataFrame, value_col: str):
    """
    Descriptive overlap validation: when the same quantity is observable in
    more than one patch, compare patch estimates.  This does not define
    observability; it diagnoses patch-context sensitivity of noisy LS fits.
    """
    rows = []
    if len(values) == 0:
        return pd.DataFrame(rows)

    for eid, g in values.groupby("interface_id"):
        if len(g) < 2:
            continue
        vals = g[value_col].to_numpy(float)
        med = float(np.median(vals))
        spread = float(np.max(np.abs(vals - med)) / max(1.0, abs(med)))
        rows.append({
            "interface_id": int(eid),
            "n_observable_patches": int(len(g)),
            "median_value": med,
            "max_relative_deviation_from_median": spread,
        })
    return pd.DataFrame(rows)


def add_solver_stability_status(df: pd.DataFrame, cfg: ProductionSolveConfig):
    if len(df) == 0:
        return df
    out = df.copy()
    q = out.production_status.eq("OBSERVABLE")
    out["numerical_status"] = np.where(
        ~q,
        "NOT_EXPORTED",
        np.where(
            out.solver_disagreement.fillna(np.inf) <= cfg.solver_agreement_tol,
            "CERTIFIED",
            "NUMERICALLY_UNSTABLE",
        ),
    )
    # A production value is exposed only if observability AND solver stability
    # both pass.  Preserve raw diagnostic estimates in separate columns.
    value_cols = [c for c in ("tension", "delta_p") if c in out.columns]
    for c in value_cols:
        out[f"{c}_production"] = np.where(
            out.numerical_status.eq("CERTIFIED"),
            out[c],
            np.nan,
        )
    return out
