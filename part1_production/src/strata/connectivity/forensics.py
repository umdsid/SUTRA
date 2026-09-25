from __future__ import annotations
import math
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely.validation import explain_validity
from scipy import ndimage
import shapely

ST4=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)

def raw_polygon_from_group(g):
    xcol="vertex_x" if "vertex_x" in g.columns else "x"
    ycol="vertex_y" if "vertex_y" in g.columns else "y"
    order=None
    for c in ("vertex_order","vertex_index","vertex_id"):
        if c in g.columns:
            order=c; break
    if order is not None:
        g=g.sort_values(order)
    xy=g[[xcol,ycol]].to_numpy(float)
    xy=xy[np.isfinite(xy).all(axis=1)]
    if len(xy)<3:
        return None,xy
    return Polygon(xy),xy

def geom_components(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type=="Polygon":
        return [geom]
    if geom.geom_type=="MultiPolygon":
        return list(geom.geoms)
    if hasattr(geom,"geoms"):
        return [g for g in geom.geoms if g.geom_type=="Polygon"]
    return []

def raster_component_count(poly, scale):
    if poly is None or poly.is_empty:
        return 0
    xmin,ymin,xmax,ymax=poly.bounds
    pad=3
    W=max(8,int(math.ceil((xmax-xmin)*scale))+2*pad+1)
    H=max(8,int(math.ceil((ymax-ymin)*scale))+2*pad+1)
    yy,xx=np.mgrid[0:H,0:W]
    px=xmin+(xx-pad+0.5)/scale
    py=ymin+(yy-pad+0.5)/scale
    inside=shapely.contains_xy(poly,px.ravel(),py.ravel()).reshape(H,W)
    _,n=ndimage.label(inside,structure=ST4)
    return int(n)

def classify_cell(g,nucleus_xy=None):
    raw,xy=raw_polygon_from_group(g)
    if raw is None:
        return {"classification":"insufficient_vertices","n_vertices":len(xy)}
    raw_valid=bool(raw.is_valid)
    repaired=raw if raw_valid else raw.buffer(0)
    comps=geom_components(repaired)
    areas=np.array([p.area for p in comps],float) if comps else np.array([],float)
    total=float(areas.sum()) if len(areas) else 0.0
    secondary=0.0
    if len(areas)>1 and total>0:
        secondary=float((total-areas.max())/total)
    counts={str(s):raster_component_count(repaired,s) for s in (1.0,2.0,4.0,8.0)}
    nuc_idx=None
    if nucleus_xy is not None and comps:
        pt=Point(float(nucleus_xy[0]),float(nucleus_xy[1]))
        hit=[i for i,p in enumerate(comps) if p.contains(pt) or p.touches(pt)]
        if hit:
            nuc_idx=hit[0]
        else:
            nuc_idx=int(np.argmin([p.distance(pt) for p in comps]))
    if len(comps)>1:
        cls="continuous_geometry_multipart"
    elif not raw_valid:
        cls="invalid_raw_polygon_repaired_to_single"
    else:
        vals=list(counts.values())
        if vals[0]>1 and any(v==1 for v in vals[1:]):
            cls="resolution_dependent_raster_split"
        elif vals[0]>1 and all(v>1 for v in vals):
            cls="persistent_raster_split_single_polygon"
        else:
            cls="single_connected_polygon"
    return {
        "classification":cls,
        "n_vertices":int(len(xy)),
        "raw_valid":raw_valid,
        "raw_validity_reason":explain_validity(raw),
        "raw_geom_type":raw.geom_type,
        "repaired_geom_type":repaired.geom_type,
        "n_polygon_components":int(len(comps)),
        "secondary_area_fraction":secondary,
        "largest_component_area_fraction":float(areas.max()/total) if total>0 else 0.0,
        "nucleus_component_index":nuc_idx,
        "raster_components_scale1":counts["1.0"],
        "raster_components_scale2":counts["2.0"],
        "raster_components_scale4":counts["4.0"],
        "raster_components_scale8":counts["8.0"],
    }
