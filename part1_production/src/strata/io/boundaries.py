from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

CELL_ID_CANDIDATES = ("cell_id", "cell", "barcode")
X_CANDIDATES = ("vertex_x", "x", "x_centroid")
Y_CANDIDATES = ("vertex_y", "y", "y_centroid")

def _pick(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"could not identify required column among {candidates}; columns={list(columns)}")

def read_boundary_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path)
    cid = _pick(df.columns, CELL_ID_CANDIDATES)
    x = _pick(df.columns, X_CANDIDATES)
    y = _pick(df.columns, Y_CANDIDATES)
    out = df[[cid, x, y]].copy()
    out.columns = ["cell_id", "x", "y"]
    out["cell_id"] = out["cell_id"].astype(str)
    return out

def polygons_from_boundary_table(df: pd.DataFrame):
    polygons = {}
    diagnostics = []
    for cell_id, g in df.groupby("cell_id", sort=False):
        xy = g[["x", "y"]].to_numpy(dtype=float)
        finite = np.isfinite(xy).all(axis=1)
        xy = xy[finite]
        if len(xy) < 3:
            polygons[cell_id] = None
            diagnostics.append({"cell_id": cell_id, "status":"INVALID", "reason":"fewer_than_3_vertices"})
            continue
        poly = Polygon(xy)
        raw_valid = bool(poly.is_valid)
        repaired = False
        if not raw_valid:
            p2 = make_valid(poly)
            repaired = True
            # Keep a polygonal result only. If MultiPolygon, preserve union geometry.
            poly = p2
        status = "PASS" if (not poly.is_empty and poly.area > 0) else "INVALID"
        diagnostics.append({
            "cell_id": cell_id,
            "status": status,
            "raw_valid": raw_valid,
            "repaired": repaired,
            "geom_type": poly.geom_type,
            "area": float(poly.area) if not poly.is_empty else 0.0,
            "perimeter": float(poly.length) if not poly.is_empty else 0.0,
            "n_boundary_vertices": int(len(xy)),
        })
        polygons[cell_id] = poly if status == "PASS" else None
    return polygons, pd.DataFrame(diagnostics)
