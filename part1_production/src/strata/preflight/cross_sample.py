from __future__ import annotations
import numpy as np


def cross_sample_audit(reports):
    out={}

    # Feature panel identity and order.
    names=[r.get("_feature_names",[]) for r in reports]
    ids=[r.get("_feature_ids",[]) for r in reports]
    same_count=len(set(len(x) for x in names))==1 if names else False
    same_names=all(x==names[0] for x in names[1:]) if names and same_count else False
    same_ids=all(x==ids[0] for x in ids[1:]) if ids and len(set(len(x) for x in ids))==1 else False
    out["gene_panel"]={
        "feature_counts":[len(x) for x in names],
        "identical_name_order":bool(same_names),
        "identical_id_order":bool(same_ids),
        "status":"PASS" if same_names else "FAIL",
    }

    # Coordinate span comparability: gross unit mismatch detection only.
    spans=[]
    for r in reports:
        c=r.get("cells",{}).get("coordinates",{})
        if c.get("available"):
            spans.append((r["sample"],c.get("x_span"),c.get("y_span")))
    vals=[max(x,y) for _,x,y in spans if x and y]
    if len(vals)>=2:
        ratio=max(vals)/max(min(vals),1e-12)
        # biological specimen sizes can differ; only flag orders-of-magnitude mismatch.
        out["coordinate_scale"]={
            "spans":spans,
            "max_span_ratio":float(ratio),
            "status":"WARN" if ratio>20 else "PASS",
            "note":"This detects gross unit mismatches only; differing specimen sizes are expected."
        }
    else:
        out["coordinate_scale"]={"status":"WARN","note":"insufficient coordinate summaries"}

    # File modality consistency.
    mods={}
    for r in reports:
        mods[r["sample"]]={
            "matrix":bool(r["files"].get("matrix")),
            "cells":bool(r["files"].get("cells")),
            "cell_boundaries":bool(r["files"].get("cell_boundaries")),
            "nucleus_boundaries":bool(r["files"].get("nucleus_boundaries")),
            "transcripts":bool(r["files"].get("transcripts")),
            "morphology":bool(r.get("morphology_files")),
        }
    out["modality_presence"]=mods
    sigs={tuple(v.values()) for v in mods.values()}
    out["modality_consistency_status"]="PASS" if len(sigs)==1 else "WARN"

    return out
