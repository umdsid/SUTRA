"""STRATA v0.7.5 Step 4: local direction-dependent norm.

For each cell x:

    alpha_x(v) = sqrt(v^T G v)
    beta_x(v)  = b_x^T v
    F_x(v)     = alpha_x(v) + beta_x(v)

where G is the certified symmetric base from Step 1 and b_x is the globally
scaled directional covector from Step 3.

Because ||b_x||_{G^-1} < 1 everywhere, F_x(v) > 0 for every v != 0.

Public manuscript-facing labels should use neutral terms such as
"direction-dependent local geometry".  Internally this is a Randers-type
Finsler construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cholesky, solve_triangular


def alpha(G, v):
    v=np.asarray(v,dtype=np.float64)
    q=float(v @ (G @ v))
    if q < -1e-10:
        raise ValueError(f"negative quadratic form {q}")
    return float(np.sqrt(max(0.0,q)))


def beta(b, v):
    return float(np.asarray(b,dtype=np.float64) @ np.asarray(v,dtype=np.float64))


def directional_norm(G, b, v):
    return alpha(G,v) + beta(b,v)


def reversal_ratio(G,b,v,eps=1e-12):
    fp=directional_norm(G,b,v)
    fm=directional_norm(G,b,-np.asarray(v,dtype=np.float64))
    if fm <= eps:
        return np.inf
    return float(fp/fm)


def dual_norm_spd(G,b):
    G=np.asarray(G,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    L=cholesky(G,lower=True,check_finite=True)
    y=solve_triangular(L,b,lower=True,check_finite=False)
    return float(np.sqrt(y@y))


def theoretical_reversal_bound(rho):
    rho=float(rho)
    if not (0 <= rho < 1):
        return np.inf
    return float((1+rho)/(1-rho))


def worst_direction(G,b):
    """Return a nonzero tangent direction proportional to G^-1 b."""
    G=np.asarray(G,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    if np.allclose(b,0):
        return np.zeros_like(b)
    return np.linalg.solve(G,b)


def certify_one_covector(G,b,tol=1e-10):
    rho=dual_norm_spd(G,b)
    if not np.isfinite(rho):
        return {"certificate_pass":False,"dual_norm":rho}

    if np.allclose(b,0):
        v=np.ones(len(b),dtype=np.float64)
    else:
        v=worst_direction(G,b)

    if np.allclose(v,0):
        v=np.ones(len(b),dtype=np.float64)

    a=alpha(G,v)
    bp=beta(b,v)
    fp=directional_norm(G,b,v)
    fm=directional_norm(G,b,-v)
    ratio=float(fp/fm) if fm>0 else np.inf
    bound=theoretical_reversal_bound(rho)

    return {
        "dual_norm":rho,
        "strict_dual_bound":bool(rho < 1.0),
        "alpha_test":a,
        "beta_test":bp,
        "F_forward_test":fp,
        "F_reverse_test":fm,
        "positive_forward":bool(fp>0),
        "positive_reverse":bool(fm>0),
        "reversal_identity_error":float(abs((fp+fm)-2*a)),
        "worst_direction_ratio":ratio,
        "theoretical_reversal_bound":bound,
        "ratio_within_bound":bool(ratio <= bound + 1e-8),
        "certificate_pass":bool(
            rho < 1.0
            and fp>0
            and fm>0
            and abs((fp+fm)-2*a) <= tol
            and ratio <= bound+1e-8
        ),
    }


def empirical_direction_statistics(G,B,dual_norms,n_random=8,seed=20260821):
    """
    Audit random tangent directions and the covector-aligned worst direction.

    Returns one row per node.  Random vectors are deterministic from a fixed
    RNG seed and are used only for diagnostics, never for construction.
    """
    G=np.asarray(G,dtype=np.float64)
    B=sparse.csr_matrix(B,dtype=np.float64)
    rng=np.random.default_rng(seed)

    d=G.shape[0]
    random_vectors=rng.normal(size=(n_random,d))
    random_vectors/=np.linalg.norm(random_vectors,axis=1,keepdims=True)

    rows=[]
    for i in range(B.shape[0]):
        b=B.getrow(i).toarray().ravel()
        rho=float(dual_norms[i])

        if np.allclose(b,0):
            worst_ratio=1.0
            worst_forward=1.0
            worst_reverse=1.0
        else:
            v=worst_direction(G,b)
            worst_forward=directional_norm(G,b,v)
            worst_reverse=directional_norm(G,b,-v)
            worst_ratio=float(worst_forward/worst_reverse)

        ratios=[]
        asym=[]
        for v in random_vectors:
            fp=directional_norm(G,b,v)
            fm=directional_norm(G,b,-v)
            if fp>0 and fm>0:
                ratios.append(max(fp/fm,fm/fp))
                asym.append(abs(fp-fm)/(fp+fm))

        rows.append({
            "cell_index":i,
            "dual_norm":rho,
            "positivity_margin":1.0-rho,
            "theoretical_reversal_bound":theoretical_reversal_bound(rho),
            "worst_direction_ratio":worst_ratio,
            "worst_direction_forward":worst_forward,
            "worst_direction_reverse":worst_reverse,
            "random_reversal_ratio_median":float(np.median(ratios)) if ratios else np.nan,
            "random_reversal_ratio_max":float(np.max(ratios)) if ratios else np.nan,
            "random_directional_asymmetry_median":float(np.median(asym)) if asym else np.nan,
            "random_directional_asymmetry_max":float(np.max(asym)) if asym else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_empirical(df):
    return {
        "n_nodes":int(len(df)),
        "fraction_directional":float((df.dual_norm>1e-12).mean()),
        "dual_norm_median":float(df.dual_norm.median()),
        "dual_norm_q90":float(df.dual_norm.quantile(.90)),
        "dual_norm_q95":float(df.dual_norm.quantile(.95)),
        "dual_norm_q99":float(df.dual_norm.quantile(.99)),
        "dual_norm_max":float(df.dual_norm.max()),
        "positivity_margin_min":float(df.positivity_margin.min()),
        "worst_direction_ratio_median":float(df.worst_direction_ratio.median()),
        "worst_direction_ratio_q90":float(df.worst_direction_ratio.quantile(.90)),
        "worst_direction_ratio_q95":float(df.worst_direction_ratio.quantile(.95)),
        "worst_direction_ratio_q99":float(df.worst_direction_ratio.quantile(.99)),
        "worst_direction_ratio_max":float(df.worst_direction_ratio.max()),
        "random_reversal_ratio_median":float(df.random_reversal_ratio_median.median()),
        "random_reversal_ratio_q95":float(df.random_reversal_ratio_max.quantile(.95)),
        "random_directional_asymmetry_median":float(df.random_directional_asymmetry_median.median()),
        "random_directional_asymmetry_q95":float(df.random_directional_asymmetry_max.quantile(.95)),
        "all_positive_margins":bool((df.positivity_margin>0).all()),
        "all_finite":bool(np.isfinite(
            df[
                [
                    "dual_norm","positivity_margin",
                    "theoretical_reversal_bound",
                    "worst_direction_ratio"
                ]
            ].to_numpy()
        ).all()),
    }
