"""STRATA v0.7.4.2 supported directional communication semantics.

Admissibility:
    communication evidence must be supported.

Geometry/statistics:
    signed directionality is preserved and never used as a merger veto.

For supported communication:
    a_ij = (s_ij - s_ji) / (s_ij + s_ji)

Then a_ji = -a_ij, with |a_ij| <= 1.

For unsupported communication:
    directionality = NA.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DirectionalCommConfig:
    support_positive_quantile: float = 0.10
    support_sensitivity_quantiles: tuple = (0.01,0.05,0.10,0.20)
    eps: float = 1e-12


def raw_support(comm_ij, comm_ji):
    return (
        np.asarray(comm_ij,dtype=np.float64)
        + np.asarray(comm_ji,dtype=np.float64)
    )


def freeze_support_floor(
    edge_rel: pd.DataFrame,
    positive_quantile: float = 0.10,
    eps: float = 1e-12,
) -> dict:
    s=raw_support(edge_rel.comm_ij,edge_rel.comm_ji)
    finite=s[np.isfinite(s)]
    positive=finite[finite>eps]
    if len(positive)==0:
        raise RuntimeError("no positive communication support exists")
    floor=max(float(np.quantile(positive,positive_quantile)),eps)
    return {
        "support_floor":floor,
        "positive_quantile":float(positive_quantile),
        "n_edges":int(len(s)),
        "n_finite":int(len(finite)),
        "n_positive":int(len(positive)),
        "n_zero_or_nearzero":int(np.sum(finite<=eps)),
        "positive_fraction":float(len(positive)/len(finite)) if len(finite) else 0.0,
        "recomputed_later":False,
    }


def support_floor_sensitivity(
    edge_rel: pd.DataFrame,
    quantiles=(0.01,0.05,0.10,0.20),
    eps: float=1e-12,
) -> pd.DataFrame:
    s=raw_support(edge_rel.comm_ij,edge_rel.comm_ji)
    finite=s[np.isfinite(s)]
    positive=finite[finite>eps]
    if len(positive)==0:
        raise RuntimeError("no positive communication support exists")

    rows=[]
    for q in quantiles:
        floor=max(float(np.quantile(positive,q)),eps)
        supported=np.isfinite(s) & (s>=floor)
        rows.append({
            "positive_quantile":float(q),
            "support_floor":float(floor),
            "supported_fraction_all_edges":float(np.mean(supported)),
            "n_supported":int(np.sum(supported)),
        })
    return pd.DataFrame(rows)


def directional_state(
    comm_ij,
    comm_ji,
    support_floor: float,
    eps: float=1e-12,
) -> pd.DataFrame:
    sij=np.asarray(comm_ij,dtype=np.float64)
    sji=np.asarray(comm_ji,dtype=np.float64)
    support=sij+sji
    supported=(
        np.isfinite(sij) & np.isfinite(sji)
        & np.isfinite(support)
        & (support>=support_floor)
    )

    signed=np.full(len(support),np.nan,dtype=np.float64)
    absolute=np.full(len(support),np.nan,dtype=np.float64)
    reciprocal=np.full(len(support),np.nan,dtype=np.float64)

    q=supported
    signed[q]=(sij[q]-sji[q])/(support[q]+eps)
    signed[q]=np.clip(signed[q],-1.0,1.0)
    absolute[q]=np.abs(signed[q])
    reciprocal[q]=1.0-absolute[q]

    return pd.DataFrame({
        "comm_support":support,
        "comm_supported":supported,
        "comm_directionality":signed,
        "comm_asymmetry":absolute,
        "comm_reciprocity_descriptive":reciprocal,
    })


def audit_directional_state(D:pd.DataFrame) -> dict:
    supported=D[D.comm_supported.astype(bool)]
    if len(supported)==0:
        return {
            "n_supported":0,
            "finite_directionality_fraction":0.0,
            "bounded_directionality":False,
            "nonzero_directionality_fraction":0.0,
            "median_abs_directionality":np.nan,
        }

    x=supported.comm_directionality.to_numpy(np.float64)
    finite=np.isfinite(x)
    xf=x[finite]
    return {
        "n_supported":int(len(supported)),
        "finite_directionality_fraction":float(np.mean(finite)),
        "bounded_directionality":bool(
            len(xf)>0 and np.all(xf>=-1.0-1e-10) and np.all(xf<=1.0+1e-10)
        ),
        "nonzero_directionality_fraction":float(
            np.mean(np.abs(xf)>1e-8)
        ) if len(xf) else 0.0,
        "median_abs_directionality":float(
            np.median(np.abs(xf))
        ) if len(xf) else np.nan,
        "q90_abs_directionality":float(
            np.quantile(np.abs(xf),.90)
        ) if len(xf) else np.nan,
    }


def block_gate_diagnostics(cand:pd.DataFrame) -> dict:
    if len(cand)==0:
        return {}
    gates=[
        "pass_molecular",
        "pass_mechanics_support",
        "pass_tension",
        "pass_delta_p",
        "pass_communication_support",
    ]
    out={}
    for g in gates:
        if g in cand:
            p=float(cand[g].astype(bool).mean())
            out[g+"_pass_fraction"]=p
            out[g+"_degenerate_always_pass"]=bool(p>=1.0-1e-12)
            out[g+"_degenerate_always_fail"]=bool(p<=1e-12)
    return out


def readiness_audit(
    level0_directional:pd.DataFrame,
    sensitivity:pd.DataFrame,
    selected_boundaries:pd.DataFrame | None=None,
) -> dict:
    d=audit_directional_state(level0_directional)

    selected_supported=True
    if selected_boundaries is not None and len(selected_boundaries):
        if "comm_supported" in selected_boundaries:
            selected_supported=bool(
                selected_boundaries.comm_supported.astype(bool).all()
            )

    sens_frac=sensitivity.supported_fraction_all_edges.to_numpy(np.float64)
    sensitivity_span=float(np.max(sens_frac)-np.min(sens_frac)) if len(sens_frac) else np.nan

    # These are structural readiness tests, not biological-effect thresholds.
    ready=(
        d["n_supported"]>0
        and d["finite_directionality_fraction"]>=0.999999
        and d["bounded_directionality"]
        and selected_supported
    )

    return {
        **d,
        "all_selected_merges_have_supported_communication":selected_supported,
        "support_floor_sensitivity_span":sensitivity_span,
        "metric_readiness_structural":bool(ready),
        "directionality_used_as_admissibility_gate":False,
        "reciprocity_used_as_admissibility_gate":False,
    }


__all__=[
    "DirectionalCommConfig",
    "raw_support",
    "freeze_support_floor",
    "support_floor_sensitivity",
    "directional_state",
    "audit_directional_state",
    "block_gate_diagnostics",
    "readiness_audit",
]
