from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

FORBIDDEN_TARGET_KEYS = {
    "target_nodes","target_node_count","desired_nodes","desired_node_count",
    "stop_at_nodes","terminal_nodes","terminal_node_count"
}

def _finite(a):
    x=np.asarray(a,float)
    return x[np.isfinite(x)]

def rel_range(a, floor=1e-12):
    x=_finite(a)
    if len(x)==0:return np.nan
    med=float(np.median(np.abs(x)))
    return float((np.max(x)-np.min(x))/max(med,floor))

def abs_range(a):
    x=_finite(a)
    if len(x)==0:return np.nan
    return float(np.max(x)-np.min(x))

def trailing_windows(series, w):
    x=np.asarray(series,float)
    if len(x)<2*w:return None,None
    return x[-2*w:-w],x[-w:]

def block_slowdown(history, cfg):
    """
    Purely causal: only preceding and recent trailing windows are used.
    speed_<block> must be a non-negative scale-local change magnitude.
    """
    w=int(cfg["history_window"])
    rows=[]
    for block in cfg["scientific_blocks"]:
        c=f"speed_{block}"
        if c not in history.columns:
            rows.append({
                "block":block,"resolved":False,"slowdown":False,
                "ratio":np.nan,"recent_median":np.nan,"previous_median":np.nan
            })
            continue
        prev,recent=trailing_windows(history[c].to_numpy(float),w)
        if prev is None:
            rows.append({
                "block":block,"resolved":False,"slowdown":False,
                "ratio":np.nan,"recent_median":np.nan,"previous_median":np.nan
            })
            continue
        p=_finite(prev);r=_finite(recent)
        if len(p)<int(cfg["minimum_finite_per_window"]) or len(r)<int(cfg["minimum_finite_per_window"]):
            rows.append({
                "block":block,"resolved":False,"slowdown":False,
                "ratio":np.nan,"recent_median":np.nan,"previous_median":np.nan
            })
            continue
        pm=float(np.median(p));rm=float(np.median(r))
        ratio=rm/max(pm,float(cfg["speed_floor"]))
        sharp=bool(np.max(r)>float(cfg["sharp_change_multiple"])*max(pm,float(cfg["speed_floor"])))
        rows.append({
            "block":block,"resolved":True,
            "slowdown":bool(ratio<=float(cfg["slowdown_ratio_max"]) and not sharp),
            "ratio":ratio,"recent_median":rm,"previous_median":pm,
            "sharp_change":sharp
        })
    return pd.DataFrame(rows)

def scalar_stability(history, cfg):
    """
    Stability gates over the most recent causal window.
    No centered smoothing and no future fill.
    """
    w=int(cfg["history_window"])
    if len(history)<w:
        return {
            "mass_stable":False,"expression_stable":False,"spatial_stable":False,
            "details":{}
        }
    h=history.iloc[-w:]
    details={}

    # Mass structure.
    mass_ok=True
    for c,tol,mode in [
        ("effective_domain_number",cfg["mass_relative_range_max"],"rel"),
        ("K80",cfg["mass_relative_range_max"],"rel"),
        ("mass_gini",cfg["gini_absolute_range_max"],"abs"),
    ]:
        if c not in h.columns:
            mass_ok=False; details[c]=np.nan; continue
        v=rel_range(h[c]) if mode=="rel" else abs_range(h[c])
        details[c]=v
        if not np.isfinite(v) or v>float(tol):mass_ok=False

    # Biology.
    expr_col="dominant_expression_coherence"
    spat_col="dominant_spatial_coherence"

    if expr_col in h.columns:
        ev=rel_range(h[expr_col])
        expr_ok=bool(np.isfinite(ev) and ev<=float(cfg["biological_relative_range_max"]))
    else:
        ev=np.nan;expr_ok=False
    if spat_col in h.columns:
        sv=rel_range(h[spat_col])
        spatial_ok=bool(np.isfinite(sv) and sv<=float(cfg["biological_relative_range_max"]))
    else:
        sv=np.nan;spatial_ok=False
    details[expr_col]=ev;details[spat_col]=sv
    return {
        "mass_stable":bool(mass_ok),
        "expression_stable":bool(expr_ok),
        "spatial_stable":bool(spatial_ok),
        "details":details
    }

def evaluate_once(history, cfg):
    """
    One prospective evaluation of the frozen stopping rule.
    """
    if len(history)==0:
        return {"candidate":False,"reason":"empty_history"}

    nodes=int(history.iloc[-1]["nodes"])
    guard=int(cfg["minimum_tessellation_guard_nodes"])

    # The guard can only prevent a scientific stop; it can never create one.
    if nodes<=guard:
        return {
            "candidate":False,
            "guard_reached":True,
            "reason":"minimum_tessellation_guard_reached_without_scientific_stop"
        }

    b=block_slowdown(history,cfg)
    resolved=b[b.resolved.astype(bool)] if len(b) else b
    nresolved=int(len(resolved))
    nslow=int(resolved.slowdown.sum()) if len(resolved) else 0
    support=float(nslow/nresolved) if nresolved else 0.0

    stable=scalar_stability(history,cfg)

    candidate=bool(
        nresolved>=int(cfg["minimum_resolved_blocks"])
        and support>=float(cfg["minimum_slow_block_fraction"])
        and stable["mass_stable"]
        and stable["expression_stable"]
        and stable["spatial_stable"]
    )
    return {
        "candidate":candidate,
        "guard_reached":False,
        "resolved_blocks":nresolved,
        "slow_blocks":nslow,
        "slow_block_fraction":support,
        "mass_stable":stable["mass_stable"],
        "expression_stable":stable["expression_stable"],
        "spatial_stable":stable["spatial_stable"],
        "block_table":b,
        "stability_details":stable["details"],
        "reason":"candidate_terminal_regime" if candidate else "continue"
    }

def evaluate_persistent(history, cfg):
    """
    Stop requires candidate=True for persistence_confirmations consecutive
    prospective evaluations. Each evaluation sees only the trajectory prefix.
    """
    need=int(cfg["persistence_confirmations"])
    if len(history)<2*int(cfg["history_window"])+need-1:
        return {
            "stop":False,"reason":"insufficient_causal_history",
            "confirmations":0
        }

    flags=[]
    evals=[]
    start=len(history)-need+1
    for end in range(start,len(history)+1):
        e=evaluate_once(history.iloc[:end].copy(),cfg)
        evals.append(e);flags.append(bool(e.get("candidate",False)))

    stop=bool(len(flags)==need and all(flags))
    return {
        "stop":stop,
        "reason":"confirmed_persistent_terminal_regime" if stop else "continue",
        "confirmations":int(sum(flags)),
        "required_confirmations":need,
        "evaluations":evals
    }

def validate_frozen_config(cfg):
    errs=[]
    keys=set(cfg)
    bad=sorted(keys & FORBIDDEN_TARGET_KEYS)
    if bad:errs.append(f"forbidden target-node keys: {bad}")

    if cfg.get("timing_semantics")!="trailing_causal_only":
        errs.append("timing_semantics must be trailing_causal_only")
    if cfg.get("future_fill_allowed") is not False:
        errs.append("future_fill_allowed must be false")
    if cfg.get("centered_smoothing_allowed") is not False:
        errs.append("centered_smoothing_allowed must be false")
    if cfg.get("guard_can_trigger_success") is not False:
        errs.append("guard_can_trigger_success must be false")

    blocks=cfg.get("scientific_blocks",[])
    if len(blocks)!=len(set(blocks)):errs.append("scientific blocks must be unique")
    if len(blocks)<10:errs.append("at least ten scientific blocks required")

    required_payload=set([
        "active_label_vector","merge_ancestry","node_state","expression",
        "go_msigdb","cellchat","mechanics_confidence_provenance","pressure",
        "topology","local_geometry","directional_state","transport","geodesics",
        "holonomy","holonomy_density","component_structure",
        "first_scale_differences","second_scale_differences"
    ])
    got=set(cfg.get("required_checkpoint_payload",[]))
    miss=sorted(required_payload-got)
    if miss:errs.append(f"missing checkpoint payload: {miss}")
    return errs

def canonical_json_hash(obj):
    s=json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
    return hashlib.sha256(s).hexdigest()

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def manifest_files(project, patterns):
    rows=[]
    seen=set()
    for pat in patterns:
        for p in project.glob(pat):
            if not p.is_file() or p in seen:continue
            seen.add(p)
            rows.append({
                "relative_path":str(p.relative_to(project)),
                "size_bytes":int(p.stat().st_size),
                "sha256":sha256_file(p)
            })
    return pd.DataFrame(rows).sort_values("relative_path") if rows else pd.DataFrame(
        columns=["relative_path","size_bytes","sha256"]
    )
