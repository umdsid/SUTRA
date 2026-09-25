from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import linalg


@dataclass(frozen=True)
class RowspaceConfig:
    rank_rcond: float = 1e-9
    observable_tol: float = 1e-8
    marginal_tol: float = 1e-6
    row_normalize: bool = True


def _row_normalized_dense(A):
    D = np.asarray(A.toarray(), dtype=np.float64)
    if D.size == 0:
        return D
    norms = np.linalg.norm(D, axis=1)
    q = norms > 0
    D[q, :] /= norms[q, None]
    return D


def rowspace_basis_qr(A, cfg: RowspaceConfig = RowspaceConfig()):
    D = _row_normalized_dense(A) if cfg.row_normalize else np.asarray(A.toarray(), float)
    nrow, nvar = D.shape

    if nvar == 0:
        return np.zeros((0, 0)), {
            "numerical_rank": 0,
            "numerical_nullity": 0,
            "rank_threshold": 0.0,
            "rdiag_max": 0.0,
        }
    if nrow == 0:
        return np.zeros((nvar, 0)), {
            "numerical_rank": 0,
            "numerical_nullity": nvar,
            "rank_threshold": 0.0,
            "rdiag_max": 0.0,
        }

    Q, R, piv = linalg.qr(
        D.T,
        mode="economic",
        pivoting=True,
        check_finite=False,
        overwrite_a=True,
    )
    diag = np.abs(np.diag(R))
    dmax = float(diag.max()) if len(diag) else 0.0
    thr = float(cfg.rank_rcond * dmax)
    rank = int(np.sum(diag > thr)) if dmax > 0 else 0
    Qr = Q[:, :rank].copy()

    return Qr, {
        "numerical_rank": rank,
        "numerical_nullity": int(nvar - rank),
        "rank_threshold": thr,
        "rdiag_max": dmax,
    }


def residual_ratio_from_basis(Qr, c):
    """
    Stable row-space residual:
        ||c - Q(Q^T c)|| / ||c||.

    Do not use sqrt(||c||^2 - ||Q^T c||^2) near zero; that expression loses
    roughly half the floating-point digits by cancellation and can convert an
    exact zero residual into O(sqrt(machine-epsilon)).
    """
    c = np.asarray(c, dtype=np.float64)
    cn = float(np.linalg.norm(c))
    if cn == 0.0:
        return 0.0
    if Qr.shape[1] == 0:
        return 1.0
    residual = c - Qr @ (Qr.T @ c)
    return float(np.linalg.norm(residual) / cn)


def classify_residual(r, cfg: RowspaceConfig):
    if r <= cfg.observable_tol:
        return "OBSERVABLE"
    if r <= cfg.marginal_tol:
        return "NUMERICALLY_MARGINAL"
    return "UNRESOLVED"


def coordinate_observability(Qr, meta, cfg: RowspaceConfig):
    """
    Individual coordinate observability.

    The leverage shortcut is retained for variables far from the threshold.
    Near the observable/marginal boundary we recompute the residual explicitly
    to avoid cancellation in sqrt(1 - leverage).
    """
    n = Qr.shape[0]
    lev = np.sum(Qr * Qr, axis=1) if Qr.shape[1] else np.zeros(n, float)
    approx = np.sqrt(np.maximum(1.0 - lev, 0.0))

    # Anything plausibly near the classification boundary is recomputed using
    # an explicit projection residual. This is cheap because only near-zero
    # candidates are touched.
    near = approx <= max(10.0 * cfg.marginal_tol, 1e-5)
    if np.any(near):
        for j in np.flatnonzero(near):
            e = np.zeros(n, dtype=np.float64)
            e[j] = 1.0
            approx[j] = residual_ratio_from_basis(Qr, e)

    rows = []
    for kind, mapping in (
        ("tension", meta["eidx"]),
        ("cell_pressure", meta["cidx"]),
        ("boundary_pressure", meta["bidx"]),
    ):
        for obj, col in mapping.items():
            r = float(approx[int(col)])
            rows.append({
                "column_index": int(col),
                "variable_class": kind,
                "object_id": int(obj),
                "rowspace_residual_ratio": r,
                "rowspace_status": classify_residual(r, cfg),
            })

    return pd.DataFrame(rows)


def pressure_contrast_observability(Qr, meta, E, cfg: RowspaceConfig):
    """
    Cell-cell pressure contrast observability using the explicit normalized
    residual of c = e_i - e_j. This removes cancellation at exact/common-gauge
    observability.
    """
    n = Qr.shape[0]
    rows = []

    for r in E.itertuples():
        if r.kind != "cell_cell":
            continue

        a = int(r.cell_i)
        b = int(r.cell_j)
        if a not in meta["cidx"] or b not in meta["cidx"]:
            continue

        ia = int(meta["cidx"][a])
        ib = int(meta["cidx"][b])

        c = np.zeros(n, dtype=np.float64)
        c[ia] = 1.0
        c[ib] = -1.0
        residual = residual_ratio_from_basis(Qr, c)

        rows.append({
            "interface_id": int(r.interface_id),
            "cell_i": a,
            "cell_j": b,
            "rowspace_residual_ratio": float(residual),
            "rowspace_status": classify_residual(float(residual), cfg),
        })

    return pd.DataFrame(rows)


def patch_rowspace_observability(A, meta, E, cfg: RowspaceConfig = RowspaceConfig()):
    Qr, rmeta = rowspace_basis_qr(A, cfg)
    V = coordinate_observability(Qr, meta, cfg)
    C = pressure_contrast_observability(Qr, meta, E, cfg)

    sm = dict(rmeta)
    for kind in ("tension", "cell_pressure", "boundary_pressure"):
        q = V.variable_class == kind
        sm[f"{kind}_n"] = int(q.sum())
        if q.any():
            sm[f"{kind}_observable_fraction"] = float(
                np.mean(V.loc[q, "rowspace_status"] == "OBSERVABLE")
            )
            sm[f"{kind}_marginal_fraction"] = float(
                np.mean(V.loc[q, "rowspace_status"] == "NUMERICALLY_MARGINAL")
            )
        else:
            sm[f"{kind}_observable_fraction"] = np.nan
            sm[f"{kind}_marginal_fraction"] = np.nan

    sm["pressure_contrast_n"] = int(len(C))
    sm["pressure_contrast_observable_fraction"] = (
        float(np.mean(C.rowspace_status == "OBSERVABLE")) if len(C) else np.nan
    )
    sm["pressure_contrast_marginal_fraction"] = (
        float(np.mean(C.rowspace_status == "NUMERICALLY_MARGINAL")) if len(C) else np.nan
    )
    return sm, V, C


def regression_confusion(old_bool, new_status):
    old = np.asarray(old_bool, dtype=bool)
    new = np.asarray(new_status, dtype=object) == "OBSERVABLE"
    return {
        "n": int(len(old)),
        "agreement_fraction": float(np.mean(old == new)) if len(old) else np.nan,
        "old_observable_new_unresolved": int(np.sum(old & ~new)),
        "old_unresolved_new_observable": int(np.sum(~old & new)),
        "both_observable": int(np.sum(old & new)),
        "both_unresolved": int(np.sum(~old & ~new)),
    }
