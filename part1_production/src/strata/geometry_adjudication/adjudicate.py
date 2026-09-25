from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.strtree import STRtree


UNASSIGNED_SENTINELS = {"4294967295", "UNASSIGNED", "unassigned", "nan", "None"}


def polygons_from_boundaries(df):
    xcol="vertex_x" if "vertex_x" in df.columns else "x"
    ycol="vertex_y" if "vertex_y" in df.columns else "y"
    order=None
    for c in ("vertex_order","vertex_index","vertex_id"):
        if c in df.columns:
            order=c; break
    out={}
    for cid,g in df.groupby("cell_id",sort=False):
        if order: g=g.sort_values(order)
        xy=g[[xcol,ycol]].to_numpy(float)
        xy=xy[np.isfinite(xy).all(axis=1)]
        if len(xy)<3: continue
        p=Polygon(xy)
        if not p.is_valid: p=p.buffer(0)
        if not p.is_empty: out[str(cid)]=p
    return out


def correct_assignment_fraction(path):
    t=pd.read_csv(path)
    if "assignment" not in t.columns:
        return None
    s=t["assignment"]
    ss=s.astype(str)
    assigned = s.notna() & ~ss.isin(UNASSIGNED_SENTINELS)
    return float(assigned.mean())


def build_overlap_distance_cost(proseg_polys, orig_polys, orig_cells,
                                max_centroid_um=20.0,
                                overlap_weight=0.75,
                                distance_weight=0.25):
    """
    Build sparse candidate costs then solve one-to-one assignment.

    Cost rewards overlap of Proseg polygon with observed Xenium core and penalizes
    centroid displacement. Candidate original cells are restricted by KDTree radius.
    """
    orig_ids=list(orig_cells.cell_id.astype(str))
    orig_xy=orig_cells[["x_centroid","y_centroid"]].to_numpy(float)
    tree=cKDTree(orig_xy)

    n=len(proseg_polys); m=len(orig_ids)
    BIG=1e6
    C=np.full((n,m),BIG,dtype=float)

    for i,p in enumerate(proseg_polys):
        cen=np.array([p.centroid.x,p.centroid.y],float)
        cand=tree.query_ball_point(cen,max_centroid_um)
        if not cand:
            _,j=tree.query(cen,k=1)
            cand=[int(j)]
        for j in cand:
            cid=orig_ids[j]
            q=orig_polys.get(cid)
            overlap=0.0
            if q is not None and not q.is_empty:
                inter=p.intersection(q).area
                denom=max(q.area,1e-12)
                overlap=min(inter/denom,1.0)
            d=float(np.linalg.norm(cen-orig_xy[j]))
            dnorm=min(d/max_centroid_um,1.0)
            C[i,j]=overlap_weight*(1-overlap)+distance_weight*dnorm
    return C, orig_ids


def hungarian_match(proseg_rows, orig_polys, orig_cells, **kwargs):
    geoms=[r["geometry"] for r in proseg_rows]
    C,orig_ids=build_overlap_distance_cost(geoms,orig_polys,orig_cells,**kwargs)
    ri,ci=linear_sum_assignment(C)
    mapping={}
    costs={}
    for i,j in zip(ri,ci):
        if C[i,j] >= 1e5:
            continue
        mapping[int(i)]=orig_ids[int(j)]
        costs[int(i)]=float(C[i,j])
    return mapping,costs


def proseg_contacts_with_epsilon(proseg_rows, mapping, epsilon):
    geoms=[r["geometry"] for r in proseg_rows]
    tree=STRtree(geoms)
    pred=set()
    for i,g in enumerate(geoms):
        search_geom=g.buffer(epsilon)
        for j in tree.query(search_geom):
            j=int(j)
            if j<=i or i not in mapping or j not in mapping:
                continue
            h=geoms[j]
            if g.distance(h) <= epsilon:
                a,b=mapping[i],mapping[j]
                if a!=b:
                    pred.add(tuple(sorted((a,b))))
    return pred


def evaluate_contact_set(pred, truth):
    tp=len(pred&truth)
    precision=tp/max(len(pred),1)
    recall=tp/max(len(truth),1)
    f1=2*precision*recall/max(precision+recall,1e-15)
    return {
        "contact_precision_vs_gateB":float(precision),
        "contact_recall_vs_gateB":float(recall),
        "contact_f1_vs_gateB":float(f1),
        "new_contact_burden":float(len(pred-truth)/max(len(pred),1)),
        "lost_gateB_fraction":float(len(truth-pred)/max(len(truth),1)),
        "n_contacts":len(pred),
    }


def conservative_selection_score(score):
    """
    Common scoring rule for all methods.
    Connectivity is intentionally excluded.
    """
    f1=float(score.get("contact_f1_vs_gateB",0))
    tx=float(score.get("transcript_capture_fraction",
                       score.get("corrected_assigned_fraction",0)) or 0)
    new=float(score.get("new_contact_burden",1))
    distort=float(score.get("median_abs_log_area_ratio",0) or 0)
    return 0.50*f1 + 0.30*tx + 0.20*(1-new) - 0.05*distort


def winner_margin(ranking):
    if len(ranking)<2: return None
    return float(ranking[0][0]-ranking[1][0])
