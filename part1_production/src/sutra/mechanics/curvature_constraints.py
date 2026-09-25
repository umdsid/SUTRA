from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from scipy import sparse


@dataclass(frozen=True)
class CurvatureConfig:
    min_points: int = 8
    min_arc_length: float = 0.5
    min_radius: float = 1.0
    max_radius: float = 1e4
    max_rel_circle_rmse: float = 0.10
    min_turning_angle_rad: float = 0.05


def _sample_linestring(line: LineString, n=64):
    if line is None or line.is_empty or line.length <= 0:
        return np.empty((0,2),float)
    nn=max(8,min(n,int(max(8,math.ceil(line.length/0.05)))))
    fs=np.linspace(0.0,1.0,nn)
    pts=[]
    for f in fs:
        p=line.interpolate(float(f),normalized=True)
        pts.append((float(p.x),float(p.y)))
    return np.asarray(pts,float)


def fit_circle_pratt_like(xy: np.ndarray):
    """
    Algebraic least-squares circle fit.
    Returns center, radius, RMSE of radial residuals.
    """
    if len(xy) < 3:
        return None
    x=xy[:,0]; y=xy[:,1]
    A=np.column_stack([2*x,2*y,np.ones(len(x))])
    b=x*x+y*y
    try:
        sol, *_ = np.linalg.lstsq(A,b,rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx,cy,c=sol
    r2=c+cx*cx+cy*cy
    if not np.isfinite(r2) or r2<=0:
        return None
    R=float(math.sqrt(r2))
    rr=np.sqrt((x-cx)**2+(y-cy)**2)
    rmse=float(np.sqrt(np.mean((rr-R)**2)))
    return float(cx),float(cy),R,rmse


def signed_curvature_from_arc(line: LineString, normal_i_to_j, cfg=CurvatureConfig()):
    xy=_sample_linestring(line)
    if len(xy)<cfg.min_points or float(line.length)<cfg.min_arc_length:
        return {
            "curvature_valid":False,"reason":"insufficient_arc_support"
        }

    fit=fit_circle_pratt_like(xy)
    if fit is None:
        return {"curvature_valid":False,"reason":"circle_fit_failed"}
    cx,cy,R,rmse=fit
    rel=rmse/max(R,1e-12)
    if not (cfg.min_radius<=R<=cfg.max_radius):
        return {"curvature_valid":False,"reason":"radius_out_of_range",
                "radius":R,"circle_rmse":rmse,"circle_rel_rmse":rel}
    if rel>cfg.max_rel_circle_rmse:
        return {"curvature_valid":False,"reason":"circle_fit_residual",
                "radius":R,"circle_rmse":rmse,"circle_rel_rmse":rel}

    # Measured turning angle as a curvature identifiability check.
    v0=xy[min(2,len(xy)-1)]-xy[0]
    v1=xy[-1]-xy[max(0,len(xy)-3)]
    n0=np.linalg.norm(v0); n1=np.linalg.norm(v1)
    if n0<=0 or n1<=0:
        return {"curvature_valid":False,"reason":"degenerate_tangent"}
    v0/=n0;v1/=n1
    dot=float(np.clip(np.dot(v0,v1),-1,1))
    turning=float(math.acos(dot))
    if turning<cfg.min_turning_angle_rad:
        return {"curvature_valid":False,"reason":"arc_too_straight",
                "radius":R,"circle_rmse":rmse,"circle_rel_rmse":rel,
                "turning_angle_rad":turning}

    mid=xy[len(xy)//2]
    radial=mid-np.array([cx,cy],float)
    nr=np.linalg.norm(radial)
    if nr<=0:
        return {"curvature_valid":False,"reason":"degenerate_radial"}
    radial/=nr
    nij=np.asarray(normal_i_to_j,float)
    nn=np.linalg.norm(nij)
    if nn<=0:
        return {"curvature_valid":False,"reason":"invalid_interface_normal"}
    nij/=nn

    # Curvature sign relative to i->j normal.
    # A circular arc whose outward radial direction aligns with i->j gets +.
    sign=1.0 if float(np.dot(radial,nij))>=0 else -1.0
    kappa=sign/R

    return {
        "curvature_valid":True,
        "reason":"PASS",
        "curvature":float(kappa),
        "radius":R,
        "circle_rmse":rmse,
        "circle_rel_rmse":rel,
        "turning_angle_rad":turning,
        "n_arc_points":int(len(xy)),
        "arc_length":float(line.length),
    }


def build_young_laplace_matrix(edges, valid_cells, curvature_table):
    """
    Row per reliable-curvature interface:
        p_i - p_j - tau_e * kappa_e = 0
    matching Δp = tau*kappa under chosen sign convention.

    Column order matches [tau_edges, pressure_cells].
    """
    cmap={c:k for k,c in enumerate(valid_cells)}
    m=len(edges); n=len(valid_cells)

    key_to_e={}
    for eidx,r in enumerate(edges.itertuples(index=False)):
        key=tuple(sorted((str(r.cell_i),str(r.cell_j))))
        key_to_e[key]=eidx

    rr=[];cc=[];vv=[];meta=[];row=0
    for r in curvature_table.itertuples(index=False):
        if not bool(r.curvature_valid):
            continue
        a,b=str(r.cell_i),str(r.cell_j)
        if a not in cmap or b not in cmap:
            continue
        key=tuple(sorted((a,b)))
        if key not in key_to_e:
            continue
        eidx=key_to_e[key]
        kappa=float(r.curvature)
        if not np.isfinite(kappa) or abs(kappa)<=0:
            continue

        # p_i - p_j - tau*kappa = 0
        rr += [row,row,row]
        cc += [eidx,m+cmap[a],m+cmap[b]]
        vv += [-kappa,1.0,-1.0]
        meta.append({
            "row":row,"cell_i":a,"cell_j":b,"edge_index":eidx,
            "curvature":kappa,
        })
        row+=1

    Y=sparse.coo_matrix((vv,(rr,cc)),shape=(row,m+n)).tocsr()
    return Y,pd.DataFrame(meta)
