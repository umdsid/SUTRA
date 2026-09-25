from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry
from shapely.geometry import GeometryCollection
from shapely.strtree import STRtree
from shapely.errors import GEOSException


@dataclass(frozen=True)
class TopologyConfig:
    min_shared_length: float = 0.25
    contact_tolerance: float = 0.25
    snap_grid: float | None = None


def _cell_centroid(poly: BaseGeometry):
    c = poly.centroid
    return np.array([float(c.x), float(c.y)], dtype=float)


def _safe_geom(g):
    """Normalize None/empty-like geometry to an empty GeometryCollection."""
    if g is None:
        return GeometryCollection()
    return g


def _safe_intersection(a: BaseGeometry, b: BaseGeometry):
    try:
        g = a.intersection(b)
    except (GEOSException, ValueError, TypeError):
        return GeometryCollection()
    return _safe_geom(g)


def _safe_buffer(g: BaseGeometry, distance: float):
    try:
        b = g.buffer(distance)
    except (GEOSException, ValueError, TypeError):
        return GeometryCollection()
    return _safe_geom(b)


def _interface_midpoint(g: BaseGeometry):
    g = _safe_geom(g)
    if g.is_empty:
        return np.array([math.nan, math.nan], dtype=float)
    try:
        if g.length > 0:
            p = g.interpolate(0.5, normalized=True)
        else:
            p = g.centroid
        return np.array([float(p.x), float(p.y)], dtype=float)
    except Exception:
        return np.array([math.nan, math.nan], dtype=float)


def _oriented_normal(ci, cj):
    d = np.asarray(cj, dtype=float) - np.asarray(ci, dtype=float)
    nrm = np.linalg.norm(d)
    if not np.isfinite(nrm) or nrm == 0:
        return np.array([math.nan, math.nan], dtype=float)
    return d / nrm


def build_observed_interfaces(
    polygons: Mapping[str, BaseGeometry | None],
    config: TopologyConfig
):
    """
    Materialize boundary-supported cell-cell interfaces.

    A pair can become an edge only if measured polygon boundaries possess
    sufficient exact or tolerance-supported shared boundary length.
    Centroid distance is never an edge criterion.
    """
    ids = []
    geoms = []
    for k, g in polygons.items():
        if g is None:
            continue
        try:
            valid_for_index = (not g.is_empty) and np.isfinite(float(g.area)) and float(g.area) > 0
        except Exception:
            valid_for_index = False
        if valid_for_index:
            ids.append(str(k))
            geoms.append(g)

    if not geoms:
        return (
            pd.DataFrame(columns=[
                "cell_i","cell_j","shared_length","midpoint_x","midpoint_y",
                "normal_i_to_j_x","normal_i_to_j_y","centroid_distance",
                "support_method"
            ]),
            pd.DataFrame(columns=["cell_id","degree","total_shared_length"]),
        )

    tree = STRtree(geoms)
    rows = []
    seen = set()

    for i, gi in enumerate(geoms):
        if config.contact_tolerance > 0:
            qgeom = _safe_buffer(gi, config.contact_tolerance)
            if qgeom.is_empty:
                qgeom = gi
        else:
            qgeom = gi

        try:
            cand_idx = tree.query(qgeom)
        except Exception:
            cand_idx = []

        for jraw in cand_idx:
            try:
                j = int(jraw)
            except Exception:
                continue
            if j <= i or j >= len(geoms):
                continue

            key = (i, j)
            if key in seen:
                continue
            seen.add(key)

            gj = geoms[j]

            # Exact measured boundary support.
            try:
                bi = _safe_geom(gi.boundary)
                bj = _safe_geom(gj.boundary)
            except Exception:
                continue

            shared = _safe_intersection(bi, bj)
            try:
                shared_len = float(shared.length)
            except Exception:
                shared_len = 0.0
            if not np.isfinite(shared_len):
                shared_len = 0.0

            method = "exact_boundary_intersection"

            # Recover near-coincident measured boundary support only.
            if shared_len < config.min_shared_length and config.contact_tolerance > 0:
                bjb = _safe_buffer(bj, config.contact_tolerance)
                bib = _safe_buffer(bi, config.contact_tolerance)
                near_i = _safe_intersection(bi, bjb) if not bjb.is_empty else GeometryCollection()
                near_j = _safe_intersection(bj, bib) if not bib.is_empty else GeometryCollection()

                try:
                    li = float(near_i.length)
                except Exception:
                    li = 0.0
                try:
                    lj = float(near_j.length)
                except Exception:
                    lj = 0.0

                if not np.isfinite(li):
                    li = 0.0
                if not np.isfinite(lj):
                    lj = 0.0

                near_len = 0.5 * (li + lj)
                if near_len >= config.min_shared_length:
                    shared_len = near_len
                    shared = near_i if li >= lj else near_j
                    method = "tolerance_boundary_support"

            if shared_len < config.min_shared_length:
                continue

            try:
                ci = _cell_centroid(gi)
                cj = _cell_centroid(gj)
            except Exception:
                continue

            mid = _interface_midpoint(shared)
            n = _oriented_normal(ci, cj)
            cd = float(np.linalg.norm(cj - ci))

            rows.append({
                "cell_i": ids[i],
                "cell_j": ids[j],
                "shared_length": float(shared_len),
                "midpoint_x": float(mid[0]),
                "midpoint_y": float(mid[1]),
                "normal_i_to_j_x": float(n[0]),
                "normal_i_to_j_y": float(n[1]),
                "centroid_distance": cd,
                "support_method": method,
            })

    edge_columns = [
        "cell_i","cell_j","shared_length","midpoint_x","midpoint_y",
        "normal_i_to_j_x","normal_i_to_j_y","centroid_distance",
        "support_method"
    ]
    edges = pd.DataFrame(rows, columns=edge_columns)

    deg = {c: 0 for c in ids}
    sl = {c: 0.0 for c in ids}
    for r in edges.itertuples(index=False):
        deg[r.cell_i] += 1
        deg[r.cell_j] += 1
        sl[r.cell_i] += float(r.shared_length)
        sl[r.cell_j] += float(r.shared_length)

    nodes = pd.DataFrame({
        "cell_id": ids,
        "degree": [deg[c] for c in ids],
        "total_shared_length": [sl[c] for c in ids],
    })
    return edges, nodes
