from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np
import pandas as pd
from scipy import ndimage


@dataclass(frozen=True)
class GeometryConfig:
    pixel_size: float = 1.0
    min_interface_pixels: int = 4
    min_curvature_points: int = 8
    curvature_cap: float = 0.5


def background_components(mask: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    bg = np.asarray(mask) == 0
    st = np.ones((3,3), dtype=np.uint8)
    lab, n = ndimage.label(bg, structure=st)
    rows = []
    H, W = bg.shape
    for k in range(1, n+1):
        ys, xs = np.nonzero(lab == k)
        if len(ys) == 0:
            continue
        touches = bool((ys == 0).any() or (ys == H-1).any() or
                       (xs == 0).any() or (xs == W-1).any())
        rows.append({
            "background_component": int(k),
            "kind": "exterior" if touches else "internal_gap",
            "pixels": int(len(ys)),
            "centroid_y": float(np.mean(ys)),
            "centroid_x": float(np.mean(xs)),
        })
    return lab.astype(np.int32), pd.DataFrame(rows)


def _interface_key(a: int, b: int, bg_comp: int | None = None):
    if a > 0 and b > 0:
        x, y = sorted((int(a), int(b)))
        return ("cell_cell", x, y, 0)
    cell = int(a if a > 0 else b)
    comp = int(bg_comp or 0)
    return ("cell_boundary", cell, 0, comp)


def _circle_curvature(xy: np.ndarray, cap: float) -> tuple[float, float]:
    # Algebraic circle fit. Return curvature magnitude and a confidence proxy.
    if len(xy) < 8:
        return 0.0, 0.0
    x = xy[:,0].astype(float)
    y = xy[:,1].astype(float)
    x0, y0 = x.mean(), y.mean()
    u, v = x-x0, y-y0
    A = np.c_[2*u, 2*v, np.ones_like(u)]
    b = u*u + v*v
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = sol
        r2 = cx*cx + cy*cy + c
        if r2 <= 1e-12:
            return 0.0, 0.0
        r = math.sqrt(r2)
        kappa = min(1.0/r, cap)
        radial = np.sqrt((u-cx)**2 + (v-cy)**2)
        cv = float(np.std(radial) / max(np.mean(radial), 1e-12))
        confidence = float(np.exp(-5.0*cv))
        return float(kappa), confidence
    except Exception:
        return 0.0, 0.0


def extract_interfaces(mask: np.ndarray, cfg: GeometryConfig):
    mask = np.asarray(mask, dtype=np.int32)
    bg_lab, bg_df = background_components(mask)
    buckets = {}

    def add_pairs(a, b, y, x, orient):
        if a == b:
            return
        if a == 0 and b == 0:
            return
        bg_comp = None
        if a == 0:
            bg_comp = int(bg_lab[y, x])
        elif b == 0:
            # neighbor coordinate depends orientation
            if orient == "h":
                bg_comp = int(bg_lab[y, x+1])
            else:
                bg_comp = int(bg_lab[y+1, x])
        key = _interface_key(int(a), int(b), bg_comp)
        # midpoint in x,y
        if orient == "h":
            pt = (x + 1.0, y + 0.5)
        else:
            pt = (x + 0.5, y + 1.0)
        buckets.setdefault(key, []).append(pt)

    H, W = mask.shape
    for y in range(H):
        row = mask[y]
        for x in range(W-1):
            add_pairs(row[x], row[x+1], y, x, "h")
    for y in range(H-1):
        for x in range(W):
            add_pairs(mask[y,x], mask[y+1,x], y, x, "v")

    rows = []
    for idx, (key, pts) in enumerate(sorted(buckets.items(), key=lambda kv: kv[0])):
        kind, ci, cj, bc = key
        xy = np.asarray(pts, float)
        if len(xy) < cfg.min_interface_pixels:
            continue
        cen = xy.mean(axis=0)
        centered = xy - cen
        if len(xy) >= 2:
            C = centered.T @ centered
            vals, vecs = np.linalg.eigh(C)
            tangent = vecs[:, int(np.argmax(vals))]
            tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
        else:
            tangent = np.array([1.0, 0.0])
        kappa, kconf = _circle_curvature(xy, cfg.curvature_cap)
        rows.append({
            "interface_id": int(len(rows)),
            "kind": kind,
            "cell_i": int(ci),
            "cell_j": int(cj) if kind == "cell_cell" else None,
            "background_component": int(bc) if kind == "cell_boundary" else None,
            "length": float(len(xy) * cfg.pixel_size),
            "centroid_x": float(cen[0]),
            "centroid_y": float(cen[1]),
            "tangent_x": float(tangent[0]),
            "tangent_y": float(tangent[1]),
            "curvature": float(kappa / max(cfg.pixel_size, 1e-12)),
            "curvature_confidence": float(kconf),
            "n_interface_pixels": int(len(xy)),
        })
    return pd.DataFrame(rows), bg_df, bg_lab


def extract_junctions(mask: np.ndarray, interfaces: pd.DataFrame, bg_lab: np.ndarray):
    # Map region-pair -> interface id.
    pair_to_e = {}
    for r in interfaces.itertuples():
        if r.kind == "cell_cell":
            pair_to_e[("c", min(r.cell_i, r.cell_j), max(r.cell_i, r.cell_j))] = r.interface_id
        else:
            pair_to_e[("b", r.cell_i, int(r.background_component))] = r.interface_id

    rows = []
    H, W = mask.shape
    seen = set()
    for y in range(H-1):
        for x in range(W-1):
            vals = [
                int(mask[y,x]), int(mask[y,x+1]),
                int(mask[y+1,x]), int(mask[y+1,x+1]),
            ]
            regs = []
            for oy, ox, v in ((0,0,vals[0]),(0,1,vals[1]),(1,0,vals[2]),(1,1,vals[3])):
                if v > 0:
                    regs.append(("c", v))
                else:
                    regs.append(("b", int(bg_lab[y+oy, x+ox])))
            uniq = sorted(set(regs))
            if len(uniq) < 3:
                continue
            # require at least two cells
            if sum(1 for q in uniq if q[0] == "c") < 2:
                continue

            incident = []
            for a,b in combinations(uniq,2):
                e = None
                if a[0] == "c" and b[0] == "c":
                    e = pair_to_e.get(("c", min(a[1],b[1]), max(a[1],b[1])))
                elif a[0] != b[0]:
                    c = a[1] if a[0] == "c" else b[1]
                    bc = a[1] if a[0] == "b" else b[1]
                    e = pair_to_e.get(("b", c, bc))
                if e is not None:
                    incident.append(int(e))
            incident = sorted(set(incident))
            if len(incident) < 3:
                continue
            key = tuple(incident)
            # Merge adjacent 2x2 detections with same interface set.
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "junction_id": int(len(rows)),
                "x": float(x+1.0),
                "y": float(y+1.0),
                "incident_interfaces": incident,
                "n_regions": int(len(uniq)),
            })
    return pd.DataFrame(rows)


def extract_cell_centroids(mask: np.ndarray) -> pd.DataFrame:
    rows = []
    maxlab = int(mask.max())
    objects = ndimage.find_objects(mask, max_label=maxlab)
    for lab in range(1, maxlab+1):
        sl = objects[lab-1] if lab-1 < len(objects) else None
        if sl is None:
            continue
        ys, xs = np.nonzero(mask[sl] == lab)
        if len(xs) == 0:
            continue
        rows.append({
            "cell_label": int(lab),
            "centroid_x": float(xs.mean() + sl[1].start),
            "centroid_y": float(ys.mean() + sl[0].start),
            "pixels": int(len(xs)),
        })
    return pd.DataFrame(rows)
