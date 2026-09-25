"""Global calibration of the raw directional covector.

One scalar kappa is shared by every cell and every specimen:

    b = kappa * b_raw.

No local clipping, specimen-specific scaling, or rank transformation is
allowed.

The default target maximum dual norm is 0.90.  It is a numerical safety margin,
not a fitted biological parameter.  Sensitivity is reported at several target
bounds so downstream claims can be checked against this choice.
"""

from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from scipy import sparse


def global_scale(global_raw_max:float,target_max:float=0.90)->float:
    if not np.isfinite(global_raw_max) or global_raw_max<=0:
        raise ValueError("global_raw_max must be positive and finite")
    if not (0<target_max<1):
        raise ValueError("target_max must lie strictly in (0,1)")
    return float(target_max/global_raw_max)


def apply_global_scale(B,kappa:float):
    if not np.isfinite(kappa) or kappa<=0:
        raise ValueError("kappa must be positive and finite")
    A=sparse.csr_matrix(B,dtype=np.float64).copy()
    A.data*=kappa
    A.eliminate_zeros()
    return A


def scaled_dual_norms(raw_dual,kappa:float):
    d=np.asarray(raw_dual,dtype=np.float64)
    return kappa*d


def reversibility_bound_from_rho(rho):
    """Worst-case F(v)/F(-v) bound for Randers norm with ||b||*=rho."""
    rho=float(rho)
    if not (0<=rho<1):
        return float("inf")
    return float((1+rho)/(1-rho))


def certify_scaled_norms(scaled,target_max,tol=1e-12):
    d=np.asarray(scaled,dtype=np.float64)
    finite=bool(np.isfinite(d).all())
    nonnegative=bool(np.all(d>=-tol))
    dmax=float(np.max(d)) if len(d) else 0.0
    strict=bool(dmax<1.0)
    target_ok=bool(dmax<=target_max+1e-10)
    return {
        "all_finite":finite,
        "all_nonnegative":nonnegative,
        "scaled_dual_norm_max":dmax,
        "strict_unit_bound":strict,
        "target_bound_satisfied":target_ok,
        "minimum_finsler_positivity_margin":float(1.0-dmax),
        "worst_case_reversibility_bound":reversibility_bound_from_rho(dmax),
        "certificate_pass":bool(
            finite and nonnegative and strict and target_ok
        ),
    }


def sensitivity_table(raw_dual_by_sample,global_raw_max,targets):
    rows=[]
    for target in targets:
        k=global_scale(global_raw_max,target)
        for sample,d in raw_dual_by_sample.items():
            x=scaled_dual_norms(d,k)
            rows.append({
                "target_global_max":float(target),
                "kappa":float(k),
                "sample":sample,
                "scaled_max":float(np.max(x)),
                "scaled_q50":float(np.quantile(x,.50)),
                "scaled_q90":float(np.quantile(x,.90)),
                "scaled_q95":float(np.quantile(x,.95)),
                "scaled_q99":float(np.quantile(x,.99)),
                "positivity_margin":float(1-np.max(x)),
                "worst_case_reversibility_bound":
                    reversibility_bound_from_rho(np.max(x)),
            })
    return pd.DataFrame(rows)


def sparse_sha256(A):
    A=sparse.csr_matrix(A,dtype=np.float64)
    h=hashlib.sha256()
    for x in (
        np.asarray(A.shape,dtype=np.int64),
        A.indptr.astype(np.int64,copy=False),
        A.indices.astype(np.int64,copy=False),
        A.data.astype(np.float64,copy=False),
    ):
        h.update(x.tobytes(order="C"))
    return h.hexdigest()
