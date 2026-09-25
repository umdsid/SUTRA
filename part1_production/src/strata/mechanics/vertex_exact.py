from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class VertexDomainConfig:
    min_cell_vertices: int = 3
    max_cell_area_rel_error: float = 1.0


def _poly_area(xy: np.ndarray) -> float:
    if len(xy) < 3:
        return math.nan
    x = xy[:,0]
    y = xy[:,1]
    return 0.5 * float(np.sum(x*np.roll(y,-1) - np.roll(x,-1)*y))


def _unique_preserve(seq):
    seen=set()
    out=[]
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_cell_vertex_cycles(
    edges: pd.DataFrame,
    junctions: pd.DataFrame,
    cells: pd.DataFrame,
    cfg: VertexDomainConfig = VertexDomainConfig(),
):
    """
    Construct an ordered junction cycle for each cell.

    The global junction coordinates are the measured-interface endpoint clusters
    already calibrated in Gate C. For each cell, all incident junctions are
    collected and ordered by polar angle about the measured Xenium cell centroid.

    Exact polygon derivatives are then taken on this reconstructed vertex network.
    """
    jxy = junctions.set_index("junction_id")[["x","y"]]
    incident = {}
    for r in edges.itertuples(index=False):
        a,b = str(r.cell_i), str(r.cell_j)
        j0,j1 = int(r.junction0), int(r.junction1)
        incident.setdefault(a, []).extend([j0,j1])
        incident.setdefault(b, []).extend([j0,j1])

    ctab = cells.copy()
    ctab["cell_id"] = ctab["cell_id"].astype(str)
    ctab = ctab.set_index("cell_id")

    cycle_rows=[]
    diag_rows=[]

    for cid, js in incident.items():
        js = _unique_preserve(js)
        if cid not in ctab.index:
            continue
        row = ctab.loc[cid]
        cx = float(row.get("x_centroid", np.nan))
        cy = float(row.get("y_centroid", np.nan))
        measured_area = float(row.get("cell_area", np.nan))

        valid_js = [j for j in js if j in jxy.index]
        if len(valid_js) < cfg.min_cell_vertices or not np.isfinite([cx,cy]).all():
            diag_rows.append({
                "cell_id":cid,
                "n_vertices":len(valid_js),
                "measured_area":measured_area,
                "reconstructed_area":math.nan,
                "relative_area_error":math.nan,
                "geometry_valid":False,
                "reason":"insufficient_vertices_or_centroid",
            })
            continue

        pts = jxy.loc[valid_js].to_numpy(float)
        ang = np.arctan2(pts[:,1]-cy, pts[:,0]-cx)
        order = np.argsort(ang)
        ordered_js = [valid_js[k] for k in order]
        ordered_xy = pts[order]

        signed_area = _poly_area(ordered_xy)
        # Enforce counterclockwise orientation.
        if np.isfinite(signed_area) and signed_area < 0:
            ordered_js = ordered_js[::-1]
            ordered_xy = ordered_xy[::-1]
            signed_area = -signed_area

        recon_area = abs(float(signed_area)) if np.isfinite(signed_area) else math.nan
        if np.isfinite(measured_area) and measured_area > 0 and np.isfinite(recon_area):
            relerr = abs(recon_area-measured_area)/measured_area
        else:
            relerr = math.nan

        geometry_valid = (
            len(ordered_js) >= cfg.min_cell_vertices
            and np.isfinite(recon_area)
            and recon_area > 0
            and np.isfinite(relerr)
            and relerr <= cfg.max_cell_area_rel_error
        )

        diag_rows.append({
            "cell_id":cid,
            "n_vertices":len(ordered_js),
            "measured_area":measured_area,
            "reconstructed_area":recon_area,
            "relative_area_error":relerr,
            "geometry_valid":bool(geometry_valid),
            "reason":"PASS" if geometry_valid else "area_mismatch_or_invalid_polygon",
        })

        if geometry_valid:
            for k,(jid,xy) in enumerate(zip(ordered_js,ordered_xy)):
                cycle_rows.append({
                    "cell_id":cid,
                    "vertex_order":k,
                    "junction_id":int(jid),
                    "x":float(xy[0]),
                    "y":float(xy[1]),
                })

    return pd.DataFrame(cycle_rows), pd.DataFrame(diag_rows)


def exact_area_derivative(prev_xy, next_xy):
    """
    For a counterclockwise polygon with area
      A = 1/2 sum_k (x_k y_{k+1} - x_{k+1} y_k),

    dA/dx_k = 1/2 (y_{k+1} - y_{k-1})
    dA/dy_k = 1/2 (x_{k-1} - x_{k+1}).
    """
    x_prev,y_prev = map(float,prev_xy)
    x_next,y_next = map(float,next_xy)
    return np.array([
        0.5*(y_next-y_prev),
        0.5*(x_prev-x_next),
    ],dtype=float)


def exact_length_derivatives(x0, x1):
    """
    For L = ||x1-x0||:
      dL/dx0 = (x0-x1)/L
      dL/dx1 = (x1-x0)/L.
    """
    x0=np.asarray(x0,float)
    x1=np.asarray(x1,float)
    d=x1-x0
    L=float(np.linalg.norm(d))
    if not np.isfinite(L) or L <= 0:
        return None,None,math.nan
    u=d/L
    return -u, u, L


def build_exact_vertex_system(
    edges: pd.DataFrame,
    junctions: pd.DataFrame,
    cycles: pd.DataFrame,
):
    """
    Assemble the exact polygonal vertex equilibrium matrix for

      H = sum_e tau_e L_e - sum_i p_i A_i

    and equations grad_{r_v} H = 0.

    Columns: all retained interface tensions followed by all retained cell pressures.
    Rows: x,y equilibrium for every retained global junction.
    """
    valid_cells = sorted(cycles["cell_id"].astype(str).unique().tolist())
    valid_set = set(valid_cells)

    # Retain only interfaces whose two cells are in the valid polygonal domain.
    e = edges[
        edges["cell_i"].astype(str).isin(valid_set)
        & edges["cell_j"].astype(str).isin(valid_set)
    ].copy().reset_index(drop=True)

    used_j = sorted(set(e["junction0"].astype(int)) | set(e["junction1"].astype(int)))
    jmap = {jid:k for k,jid in enumerate(used_j)}
    cmap = {cid:k for k,cid in enumerate(valid_cells)}
    m=len(e); n=len(valid_cells)

    jxy = junctions.set_index("junction_id")[["x","y"]]

    rr=[];cc=[];vv=[]
    def add(row,col,val):
        rr.append(int(row)); cc.append(int(col)); vv.append(float(val))

    # Exact interface-length gradients.
    retained_edge_rows=[]
    for eidx,r in enumerate(e.itertuples(index=False)):
        j0,j1=int(r.junction0),int(r.junction1)
        if j0 not in jmap or j1 not in jmap or j0 not in jxy.index or j1 not in jxy.index:
            continue
        x0=jxy.loc[j0].to_numpy(float)
        x1=jxy.loc[j1].to_numpy(float)
        g0,g1,L=exact_length_derivatives(x0,x1)
        if g0 is None:
            continue
        for comp in (0,1):
            add(2*jmap[j0]+comp, eidx, g0[comp])
            add(2*jmap[j1]+comp, eidx, g1[comp])
        retained_edge_rows.append((eidx,L))

    # Exact polygon-area gradients.
    for cid,g in cycles.groupby("cell_id",sort=False):
        cid=str(cid)
        if cid not in cmap:
            continue
        g=g.sort_values("vertex_order")
        js=g["junction_id"].astype(int).tolist()
        xy=g[["x","y"]].to_numpy(float)
        q=len(js)
        if q<3:
            continue
        pcol=m+cmap[cid]
        for k,jid in enumerate(js):
            if jid not in jmap:
                continue
            dA=exact_area_derivative(xy[(k-1)%q],xy[(k+1)%q])
            # H contains -p_i A_i.
            for comp in (0,1):
                add(2*jmap[jid]+comp,pcol,-dA[comp])

    A=sparse.coo_matrix(
        (vv,(rr,cc)),
        shape=(2*len(used_j),m+n)
    ).tocsr()

    return A,e,valid_cells,used_j
