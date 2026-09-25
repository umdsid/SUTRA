from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy import ndimage
from shapely.geometry import Polygon, Point
from shapely.prepared import prep


@dataclass(frozen=True)
class GapAuditConfig:
    raster_max_pixels: int = 8_000_000
    tissue_intensity_quantile: float = 0.35
    morphology_downsample_max_dim: int = 2500
    min_internal_hole_pixels: int = 8


def discover_morphology_files(sample_root: Path):
    pats = [
        "*morphology*.ome.tif", "*morphology*.ome.tiff",
        "*morphology*.tif", "*morphology*.tiff",
        "*he*.tif", "*h&e*.tif", "*dapi*.tif",
        "*.ome.tif", "*.ome.tiff",
    ]
    seen=set(); hits=[]
    for pat in pats:
        for p in sample_root.rglob(pat):
            if p.is_file() and p not in seen:
                seen.add(p); hits.append(p)
    def score(p):
        s=p.name.lower()
        return (
            0 if "morphology" in s else 1,
            0 if "focus" in s else 1,
            0 if ("he" in s or "h&e" in s or "dapi" in s) else 1,
            len(s),
        )
    return sorted(hits,key=score)


def _bounds_from_boundary_df(df):
    xcol="vertex_x" if "vertex_x" in df.columns else ("x" if "x" in df.columns else None)
    ycol="vertex_y" if "vertex_y" in df.columns else ("y" if "y" in df.columns else None)
    if xcol is None or ycol is None:
        raise ValueError("boundary table lacks x/y coordinates")
    return (
        float(df[xcol].min()),float(df[ycol].min()),
        float(df[xcol].max()),float(df[ycol].max())
    ),xcol,ycol


def polygons_from_boundary_table_generic(df):
    if "cell_id" not in df.columns:
        raise ValueError("boundary table lacks cell_id")
    bounds,xcol,ycol=_bounds_from_boundary_df(df)
    order_col=None
    for c in ("vertex_order","vertex_index","vertex_id"):
        if c in df.columns:
            order_col=c;break

    polys={}
    for cell,g in df.groupby("cell_id",sort=False):
        if order_col is not None:
            g=g.sort_values(order_col)
        xy=g[[xcol,ycol]].to_numpy(float)
        if len(xy)<3:
            continue
        try:
            poly=Polygon(xy)
            if not poly.is_valid:
                poly=poly.buffer(0)
            if poly.is_empty:
                continue
            polys[str(cell)]=poly
        except Exception:
            continue
    return polys,bounds


def rasterize_polygons(polys,bounds,max_pixels):
    """
    Rasterize cell polygons without matplotlib.

    Uses Shapely prepared geometries over each polygon's pixel bounding box.
    This is slower than matplotlib.path but dependency-safe for the native
    STRATA environment and adequate for the one-off integrity audit.
    """
    xmin,ymin,xmax,ymax=bounds
    w=max(xmax-xmin,1e-9);h=max(ymax-ymin,1e-9)
    scale=math.sqrt(max_pixels/max(w*h,1e-12))
    scale=min(max(scale,0.1),4.0)
    W=max(8,int(math.ceil(w*scale))+3)
    H=max(8,int(math.ceil(h*scale))+3)

    mask=np.zeros((H,W),dtype=bool)

    for poly in polys.values():
        try:
            minx,miny,maxx,maxy=poly.bounds
            bx0=max(0,int(math.floor((minx-xmin)*scale))+1)
            bx1=min(W,int(math.ceil((maxx-xmin)*scale))+2)
            by0=max(0,int(math.floor((miny-ymin)*scale))+1)
            by1=min(H,int(math.ceil((maxy-ymin)*scale))+2)
            if bx1<=bx0 or by1<=by0:
                continue

            ppoly=prep(poly)

            # Pixel-center coordinates mapped back to physical space.
            for iy in range(by0,by1):
                py=ymin+(iy-1+0.5)/scale
                xs=[]
                for ix in range(bx0,bx1):
                    px=xmin+(ix-1+0.5)/scale
                    if ppoly.contains(Point(px,py)) or ppoly.covers(Point(px,py)):
                        xs.append(ix)
                if xs:
                    mask[iy,xs]=True
        except Exception:
            continue

    return mask,{"scale":scale,"shape":[H,W],"bounds":[xmin,ymin,xmax,ymax]}


def internal_gap_metrics(cell_mask):
    occupied=cell_mask.astype(bool)
    filled=ndimage.binary_fill_holes(occupied)
    internal=filled & ~occupied

    lab,n=ndimage.label(internal)
    sizes=np.bincount(lab.ravel()) if n else np.array([0])
    hole_sizes=sizes[1:] if len(sizes)>1 else np.array([],dtype=int)

    return {
        "cell_coverage_fraction_of_filled_envelope":float(
            occupied.sum()/max(filled.sum(),1)
        ),
        "internal_gap_fraction_of_filled_envelope":float(
            internal.sum()/max(filled.sum(),1)
        ),
        "n_internal_gap_components":int(n),
        "largest_internal_gap_pixels":int(hole_sizes.max()) if len(hole_sizes) else 0,
        "median_internal_gap_pixels":float(np.median(hole_sizes)) if len(hole_sizes) else 0.0,
    },filled,internal


def load_morphology(path: Path,max_dim=2500):
    import tifffile
    arr=np.asarray(tifffile.imread(path))
    while arr.ndim>2:
        arr=np.nanmax(arr,axis=0)
    arr=arr.astype(np.float32)
    stride=max(1,int(math.ceil(max(arr.shape)/max_dim)))
    arr=arr[::stride,::stride]
    finite=np.isfinite(arr)
    if finite.any():
        lo=float(np.nanpercentile(arr[finite],1))
        hi=float(np.nanpercentile(arr[finite],99))
        if hi>lo:
            arr=np.clip((arr-lo)/(hi-lo),0,1)
        else:
            arr=np.zeros_like(arr)
    return arr,stride


def morphology_tissue_mask(img,q=0.35):
    x=np.asarray(img,float)
    gx=ndimage.sobel(x,axis=1,mode="nearest")
    gy=ndimage.sobel(x,axis=0,mode="nearest")
    g=np.hypot(gx,gy)
    sm=ndimage.gaussian_filter(g,1.5)
    thr=float(np.quantile(sm,q))
    tissue=sm>thr
    tissue=ndimage.binary_closing(tissue,iterations=3)
    tissue=ndimage.binary_fill_holes(tissue)
    return tissue,thr


def resize_bool(mask,shape):
    zoom=(shape[0]/mask.shape[0],shape[1]/mask.shape[1])
    out=ndimage.zoom(mask.astype(np.uint8),zoom=zoom,order=0)
    return out[:shape[0],:shape[1]].astype(bool)


def compare_to_morphology(cell_mask,morphology_img,q):
    tissue_img,thr=morphology_tissue_mask(morphology_img,q)
    cm=resize_bool(cell_mask,tissue_img.shape)

    hull=ndimage.binary_fill_holes(cm)
    tissue_local=tissue_img & hull
    unassigned=tissue_local & ~cm

    return {
        "morphology_threshold":thr,
        "cell_coverage_fraction_of_morphology_tissue_support":float(
            (cm&tissue_local).sum()/max(tissue_local.sum(),1)
        ),
        "unassigned_fraction_of_morphology_tissue_support":float(
            unassigned.sum()/max(tissue_local.sum(),1)
        ),
        "morphology_tissue_fraction_within_cell_envelope":float(
            tissue_local.sum()/max(hull.sum(),1)
        ),
    },tissue_img,unassigned
