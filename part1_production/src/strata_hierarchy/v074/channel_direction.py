"""Channel-resolved directed-communication audit for STRATA v0.7.4.3.

This module does not alter the hierarchy.

For interaction channel r on measured contact i--j:

    f_r(i->j) = sqrt(y_i,sender * y_j,receiver)
    f_r(j->i) = sqrt(y_j,sender * y_i,receiver)

    delta_r = f_r(i->j) - f_r(j->i)
    weight_r = f_r(i->j) + f_r(j->i)

From all supported channels:

    signed_net = sum(delta_r) / sum(weight_r)
    net_asymmetry = |sum(delta_r)| / sum(weight_r)
    directional_content = sum(|delta_r|) / sum(weight_r)
    cancellation = 1 - |sum(delta_r)| / sum(|delta_r|)

Thus directional_content >= net_asymmetry.  Equality means no cancellation
between directional channels.  High cancellation means opposing directional
programs were hidden by scalar reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChannelAuditConfig:
    eps: float = 1e-12
    zero_tol: float = 1e-10


def _safe_ratio(num,den,eps=1e-12):
    num=np.asarray(num,dtype=np.float64)
    den=np.asarray(den,dtype=np.float64)
    out=np.full(np.broadcast(num,den).shape,np.nan,dtype=np.float64)
    q=np.isfinite(num)&np.isfinite(den)&(den>eps)
    out[q]=num[q]/den[q]
    return out


def channel_fluxes(
    Y: np.ndarray,
    edges: pd.DataFrame,
    genes: tuple[str,...],
    registry: pd.DataFrame,
):
    """Yield one channel at a time without materializing edge x channel cube."""
    idx={str(g):k for k,g in enumerate(genes)}

    reg=registry.copy()
    if "panel_supported" in reg.columns:
        reg=reg[reg.panel_supported.astype(bool)]
    reg=reg[
        reg.sender_gene.astype(str).isin(idx)
        & reg.receiver_gene.astype(str).isin(idx)
        & reg.kind.astype(str).isin(["contact","diffusible"])
    ].copy()

    i=edges.cell_i_index.to_numpy(np.int64)
    j=edges.cell_j_index.to_numpy(np.int64)

    for ridx,r in reg.reset_index(drop=True).iterrows():
        s=idx[str(r.sender_gene)]
        t=idx[str(r.receiver_gene)]

        fij=np.sqrt(
            np.maximum(0.0,Y[i,s].astype(np.float64))
            * np.maximum(0.0,Y[j,t].astype(np.float64))
        )
        fji=np.sqrt(
            np.maximum(0.0,Y[j,s].astype(np.float64))
            * np.maximum(0.0,Y[i,t].astype(np.float64))
        )

        yield {
            "channel_index":int(ridx),
            "sender_gene":str(r.sender_gene),
            "receiver_gene":str(r.receiver_gene),
            "kind":str(r.kind),
            "forward":fij,
            "reverse":fji,
            "delta":fij-fji,
            "weight":fij+fji,
        }


def channel_resolved_edge_metrics(
    Y: np.ndarray,
    edges: pd.DataFrame,
    genes: tuple[str,...],
    registry: pd.DataFrame,
    cfg: ChannelAuditConfig=ChannelAuditConfig(),
):
    """
    Return per-edge aggregate metrics plus per-channel summary.

    No hierarchy state changes are made.
    """
    n=len(edges)
    sum_delta=np.zeros(n,dtype=np.float64)
    sum_abs_delta=np.zeros(n,dtype=np.float64)
    sum_weight=np.zeros(n,dtype=np.float64)

    by_kind={
        k:{
            "sum_delta":np.zeros(n,dtype=np.float64),
            "sum_abs_delta":np.zeros(n,dtype=np.float64),
            "sum_weight":np.zeros(n,dtype=np.float64),
        }
        for k in ("contact","diffusible")
    }

    rows=[]
    n_channels=0

    for ch in channel_fluxes(Y,edges,genes,registry):
        n_channels+=1
        d=ch["delta"]
        w=ch["weight"]
        ad=np.abs(d)

        sum_delta+=d
        sum_abs_delta+=ad
        sum_weight+=w

        bk=by_kind[ch["kind"]]
        bk["sum_delta"]+=d
        bk["sum_abs_delta"]+=ad
        bk["sum_weight"]+=w

        active=w>cfg.eps
        directed=ad>cfg.zero_tol

        rows.append({
            "channel_index":ch["channel_index"],
            "sender_gene":ch["sender_gene"],
            "receiver_gene":ch["receiver_gene"],
            "kind":ch["kind"],
            "n_edges":int(n),
            "n_active_edges":int(active.sum()),
            "active_edge_fraction":float(active.mean()),
            "n_directional_edges":int(directed.sum()),
            "directional_edge_fraction":float(directed.mean()),
            "total_forward":float(np.sum(ch["forward"])),
            "total_reverse":float(np.sum(ch["reverse"])),
            "total_support":float(np.sum(w)),
            "signed_flux_sum":float(np.sum(d)),
            "absolute_flux_sum":float(np.sum(ad)),
            "global_signed_directionality":float(
                np.sum(d)/(np.sum(w)+cfg.eps)
            ) if np.sum(w)>cfg.eps else np.nan,
            "global_directional_content":float(
                np.sum(ad)/(np.sum(w)+cfg.eps)
            ) if np.sum(w)>cfg.eps else np.nan,
        })

    if n_channels==0:
        raise RuntimeError("no panel-supported communication channels available")

    signed_net=_safe_ratio(sum_delta,sum_weight,cfg.eps)
    net_asym=_safe_ratio(np.abs(sum_delta),sum_weight,cfg.eps)
    directional_content=_safe_ratio(sum_abs_delta,sum_weight,cfg.eps)

    cancellation=np.full(n,np.nan,dtype=np.float64)
    q=sum_abs_delta>cfg.eps
    cancellation[q]=1.0-np.abs(sum_delta[q])/(sum_abs_delta[q]+cfg.eps)
    cancellation[q]=np.clip(cancellation[q],0.0,1.0)

    E=pd.DataFrame({
        "channel_support_sum":sum_weight,
        "channel_signed_net":signed_net,
        "channel_net_asymmetry":net_asym,
        "channel_directional_content":directional_content,
        "channel_cancellation":cancellation,
        "n_channels":int(n_channels),
    })

    for kind,bk in by_kind.items():
        sw=bk["sum_weight"]
        sd=bk["sum_delta"]
        sa=bk["sum_abs_delta"]

        E[f"{kind}_support_sum"]=sw
        E[f"{kind}_signed_net"]=_safe_ratio(sd,sw,cfg.eps)
        E[f"{kind}_net_asymmetry"]=_safe_ratio(np.abs(sd),sw,cfg.eps)
        E[f"{kind}_directional_content"]=_safe_ratio(sa,sw,cfg.eps)

        can=np.full(n,np.nan,dtype=np.float64)
        qq=sa>cfg.eps
        can[qq]=1.0-np.abs(sd[qq])/(sa[qq]+cfg.eps)
        can[qq]=np.clip(can[qq],0.0,1.0)
        E[f"{kind}_cancellation"]=can

    return E,pd.DataFrame(rows)


def compare_scalar_and_channel(
    scalar_signed,
    scalar_supported,
    channel_metrics: pd.DataFrame,
    cfg: ChannelAuditConfig=ChannelAuditConfig(),
) -> pd.DataFrame:
    """
    Compare the previous scalar direction with channel-resolved directionality.
    """
    s=np.asarray(scalar_signed,dtype=np.float64)
    supported=np.asarray(scalar_supported,dtype=bool)

    ch=channel_metrics.channel_directional_content.to_numpy(np.float64)
    net=channel_metrics.channel_net_asymmetry.to_numpy(np.float64)
    can=channel_metrics.channel_cancellation.to_numpy(np.float64)

    hidden=(
        supported
        & np.isfinite(s)
        & (np.abs(s)<=cfg.zero_tol)
        & np.isfinite(ch)
        & (ch>cfg.zero_tol)
    )
    cancellation_present=(
        supported
        & np.isfinite(can)
        & (can>cfg.zero_tol)
    )

    return pd.DataFrame({
        "scalar_supported":supported,
        "scalar_signed_directionality":s,
        "scalar_abs_directionality":np.abs(s),
        "channel_directional_content":ch,
        "channel_net_asymmetry":net,
        "channel_cancellation":can,
        "hidden_directionality":hidden,
        "cancellation_present":cancellation_present,
    })


def audit_summary(
    comparison:pd.DataFrame,
    edge_metrics:pd.DataFrame,
    cfg:ChannelAuditConfig=ChannelAuditConfig(),
):
    q=comparison.scalar_supported.astype(bool)
    C=comparison[q].copy()

    if len(C)==0:
        raise RuntimeError("no scalar-supported edges available for channel audit")

    ch=C.channel_directional_content.to_numpy(np.float64)
    net=C.channel_net_asymmetry.to_numpy(np.float64)
    can=C.channel_cancellation.to_numpy(np.float64)

    finite_ch=np.isfinite(ch)
    finite_net=np.isfinite(net)

    dominance_ok=bool(
        np.all(
            ch[finite_ch & finite_net]
            + 1e-10
            >= net[finite_ch & finite_net]
        )
    )

    return {
        "n_scalar_supported_edges":int(len(C)),
        "finite_channel_directional_content_fraction":float(np.mean(finite_ch)),
        "directional_content_dominates_net_asymmetry":dominance_ok,
        "hidden_directionality_fraction_supported":float(
            C.hidden_directionality.mean()
        ),
        "cancellation_present_fraction_supported":float(
            C.cancellation_present.mean()
        ),
        "channel_directional_content_median":float(np.nanmedian(ch)),
        "channel_directional_content_q75":float(np.nanquantile(ch,.75)),
        "channel_directional_content_q90":float(np.nanquantile(ch,.90)),
        "channel_directional_content_q95":float(np.nanquantile(ch,.95)),
        "channel_net_asymmetry_median":float(np.nanmedian(net)),
        "channel_net_asymmetry_q90":float(np.nanquantile(net,.90)),
        "channel_cancellation_median_defined":float(np.nanmedian(can))
            if np.isfinite(can).any() else np.nan,
        "channel_cancellation_q90_defined":float(np.nanquantile(can,.90))
            if np.isfinite(can).any() else np.nan,
        "scalar_zero_but_channel_nonzero_edges":int(
            C.hidden_directionality.sum()
        ),
        "structural_checks_pass":bool(
            np.mean(finite_ch)>=0.999999
            and dominance_ok
        ),
    }


def choose_directional_representation(summary:dict) -> dict:
    """
    Conservative representation choice for the next local metric.

    Channel-resolved odd information is always at least as informative as the
    scalar net when the structural checks pass.  The audit reports whether
    cancellation materially exists; it does not invent a biological cutoff.
    """
    if not summary["structural_checks_pass"]:
        return {
            "status":"HOLD",
            "recommended_representation":None,
            "reason":"channel-resolved structural checks failed",
        }

    return {
        "status":"PASS",
        "recommended_representation":"channel_resolved_signed_contributions",
        "retain_scalar_net_for":"descriptive comparison",
        "retain_directional_content_for":"magnitude/statistical diagnostics",
        "retain_cancellation_for":"opposing-program diagnostics",
        "reason":(
            "channel-resolved signed contributions preserve net direction "
            "and cannot lose opposing directional programs through scalar cancellation"
        ),
    }


__all__=[
    "ChannelAuditConfig",
    "channel_fluxes",
    "channel_resolved_edge_metrics",
    "compare_scalar_and_channel",
    "audit_summary",
    "choose_directional_representation",
]
