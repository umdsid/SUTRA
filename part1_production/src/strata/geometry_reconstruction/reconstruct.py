from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy import ndimage
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree


@dataclass(frozen=True)
class ReconstructionConfig:
    max_assignment_distance: float = 6.0
    neighbor_candidate_hops: int = 1
    raster_max_pixels: int = 10_000_000
    morphology_quantile: float = 0.35
    preserve_observed_polygons: bool = True


def build_adjacency(edges: pd.DataFrame):
    adj = {}
    for r in edges.itertuples(index=False):
        a, b = str(r.cell_i), str(r.cell_j)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def polygons_from_boundary_table(df: pd.DataFrame):
    xcol = "vertex_x" if "vertex_x" in df.columns else "x"
    ycol = "vertex_y" if "vertex_y" in df.columns else "y"
    order_col = None
    for c in ("vertex_order", "vertex_index", "vertex_id"):
        if c in df.columns:
            order_col = c
            break

    polys = {}
    for cid, g in df.groupby("cell_id", sort=False):
        if order_col:
            g = g.sort_values(order_col)
        xy = g[[xcol, ycol]].to_numpy(float)
        if len(xy) < 3:
            continue
        try:
            p = Polygon(xy)
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_empty:
                continue
            polys[str(cid)] = p
        except Exception:
            continue
    return polys


def _bounds(polys):
    b = np.array([p.bounds for p in polys.values()], float)
    return (
        float(np.min(b[:,0])), float(np.min(b[:,1])),
        float(np.max(b[:,2])), float(np.max(b[:,3])),
    )


def _raster_geometry(polys, max_pixels):
    xmin,ymin,xmax,ymax = _bounds(polys)
    w=max(xmax-xmin,1e-9); h=max(ymax-ymin,1e-9)
    scale=min(max(math.sqrt(max_pixels/(w*h)),0.2),4.0)
    W=max(8,int(math.ceil(w*scale))+3)
    H=max(8,int(math.ceil(h*scale))+3)

    labels=np.zeros((H,W),np.int32)
    cell_ids=list(polys.keys())
    id_to_label={cid:i+1 for i,cid in enumerate(cell_ids)}
    label_to_id={i+1:cid for i,cid in enumerate(cell_ids)}

    yy,xx=np.mgrid[0:H,0:W]
    px=xmin+(xx-1+0.5)/scale
    py=ymin+(yy-1+0.5)/scale

    # Efficient per-polygon bounding-box fill.
    for cid,p in polys.items():
        lab=id_to_label[cid]
        bx0=max(0,int(math.floor((p.bounds[0]-xmin)*scale))+1)
        bx1=min(W,int(math.ceil((p.bounds[2]-xmin)*scale))+2)
        by0=max(0,int(math.floor((p.bounds[1]-ymin)*scale))+1)
        by1=min(H,int(math.ceil((p.bounds[3]-ymin)*scale))+2)
        if bx1<=bx0 or by1<=by0:
            continue
        subx=px[by0:by1,bx0:bx1]
        suby=py[by0:by1,bx0:bx1]
        pts=np.c_[subx.ravel(),suby.ravel()]
        # Shapely 2 vectorized contains if available.
        try:
            import shapely
            inside=shapely.contains_xy(p,pts[:,0],pts[:,1]).reshape(subx.shape)
        except Exception:
            inside=np.array([p.covers(Point(x,y)) for x,y in pts],bool).reshape(subx.shape)
        labels[by0:by1,bx0:bx1][inside]=lab

    meta={
        "bounds":[xmin,ymin,xmax,ymax],
        "scale":float(scale),
        "shape":[H,W],
        "cell_ids":cell_ids,
        "label_to_cell_id":label_to_id,
        "cell_id_to_label":id_to_label,
    }
    return labels,meta


def morphology_support_from_image(img, shape, quantile=0.35):
    """
    Conservative image-bearing tissue heuristic. This is not histological
    classification; it only defines where geometry may be reconstructed.
    """
    x=np.asarray(img,float)
    while x.ndim>2:
        x=np.nanmax(x,axis=0)
    finite=np.isfinite(x)
    if finite.any():
        lo=float(np.nanpercentile(x[finite],1))
        hi=float(np.nanpercentile(x[finite],99))
        if hi>lo:
            x=np.clip((x-lo)/(hi-lo),0,1)
    gx=ndimage.sobel(x,axis=1,mode="nearest")
    gy=ndimage.sobel(x,axis=0,mode="nearest")
    g=ndimage.gaussian_filter(np.hypot(gx,gy),1.5)
    thr=float(np.quantile(g,quantile))
    tissue=g>thr
    tissue=ndimage.binary_closing(tissue,iterations=3)
    tissue=ndimage.binary_fill_holes(tissue)

    zoom=(shape[0]/tissue.shape[0],shape[1]/tissue.shape[1])
    out=ndimage.zoom(tissue.astype(np.uint8),zoom=zoom,order=0)
    return out[:shape[0],:shape[1]].astype(bool),thr


def nucleus_support_from_boundaries(nucleus_polys, raster_meta):
    if not nucleus_polys:
        return None
    xmin,ymin,xmax,ymax=raster_meta["bounds"]
    scale=raster_meta["scale"]
    H,W=raster_meta["shape"]
    out=np.zeros((H,W),bool)
    for p in nucleus_polys.values():
        bx0=max(0,int(math.floor((p.bounds[0]-xmin)*scale))+1)
        bx1=min(W,int(math.ceil((p.bounds[2]-xmin)*scale))+2)
        by0=max(0,int(math.floor((p.bounds[1]-ymin)*scale))+1)
        by1=min(H,int(math.ceil((p.bounds[3]-ymin)*scale))+2)
        if bx1<=bx0 or by1<=by0:
            continue
        yy,xx=np.mgrid[by0:by1,bx0:bx1]
        px=xmin+(xx-1+0.5)/scale
        py=ymin+(yy-1+0.5)/scale
        pts=np.c_[px.ravel(),py.ravel()]
        try:
            import shapely
            inside=shapely.contains_xy(p,pts[:,0],pts[:,1]).reshape(px.shape)
        except Exception:
            inside=np.array([p.covers(Point(x,y)) for x,y in pts],bool).reshape(px.shape)
        out[by0:by1,bx0:bx1] |= inside
    return out


def transcript_density_mask(transcripts: pd.DataFrame, raster_meta):
    """
    Rasterize transcript coordinates. Supports common Xenium coordinate names.
    """
    xcol=None; ycol=None
    for c in ("x_location","x","transcript_x"):
        if c in transcripts.columns:
            xcol=c;break
    for c in ("y_location","y","transcript_y"):
        if c in transcripts.columns:
            ycol=c;break
    if xcol is None or ycol is None:
        return None

    xmin,ymin,xmax,ymax=raster_meta["bounds"]
    scale=raster_meta["scale"]
    H,W=raster_meta["shape"]

    x=transcripts[xcol].to_numpy(float)
    y=transcripts[ycol].to_numpy(float)
    ix=np.floor((x-xmin)*scale+1).astype(int)
    iy=np.floor((y-ymin)*scale+1).astype(int)
    ok=(ix>=0)&(ix<W)&(iy>=0)&(iy<H)
    hist=np.zeros((H,W),np.float32)
    np.add.at(hist,(iy[ok],ix[ok]),1.0)
    # diffuse locally to represent measured transcript support
    sm=ndimage.gaussian_filter(hist,1.5)
    return sm


def reconstruct_partition(
    cell_polys,
    certified_edges,
    morphology_support,
    nucleus_support,
    transcript_density,
    cfg=ReconstructionConfig(),
):
    """
    Construct a mechanics-grade partition without overwriting observed geometry.

    Observed labeled pixels are immutable.
    Candidate pixels must be morphology-supported and currently unassigned.
    Assignment uses nearest observed cell territory, but only if that cell is
    topologically compatible with local observed neighbors through Gate-B.
    """
    labels,meta=_raster_geometry(cell_polys,cfg.raster_max_pixels)
    H,W=labels.shape
    observed=labels>0

    if morphology_support is None:
        tissue=ndimage.binary_fill_holes(observed)
    else:
        tissue=np.asarray(morphology_support,bool)
        if tissue.shape!=labels.shape:
            raise ValueError("morphology support shape mismatch")

    candidate=tissue & (~observed)

    # Evidence layers only control eligibility, not cell identity.
    evidence=np.zeros_like(labels,dtype=np.float32)
    evidence += morphology_support.astype(np.float32) if morphology_support is not None else 1.0
    if nucleus_support is not None:
        evidence += 2.0*nucleus_support.astype(np.float32)
    if transcript_density is not None:
        td=transcript_density.astype(np.float32)
        if np.max(td)>0:
            td=td/np.max(td)
        evidence += td

    # nearest observed label for every pixel
    _,inds=ndimage.distance_transform_edt(~observed,return_indices=True)
    nearest=labels[tuple(inds)]

    # physical distance to observed territory
    dist_pix=ndimage.distance_transform_edt(~observed)
    dist_phys=dist_pix/max(meta["scale"],1e-12)

    adj=build_adjacency(certified_edges)
    label_to_id=meta["label_to_cell_id"]
    id_to_label=meta["cell_id_to_label"]

    # Determine local observed neighboring labels by 1-pixel dilation.
    dil=ndimage.grey_dilation(labels,size=(3,3))
    ero=ndimage.grey_erosion(labels,size=(3,3))
    local_nonzero=((dil>0)|(ero>0))

    projected=labels.copy()
    accepted=np.zeros_like(candidate,bool)
    rejected_distance=np.zeros_like(candidate,bool)
    rejected_topology=np.zeros_like(candidate,bool)

    ys,xs=np.nonzero(candidate)
    for y,x in zip(ys,xs):
        lab=int(nearest[y,x])
        if lab<=0:
            continue
        if dist_phys[y,x]>cfg.max_assignment_distance:
            rejected_distance[y,x]=True
            continue

        cid=label_to_id[lab]

        # Collect observed labels in a small local neighborhood.
        y0=max(0,y-2);y1=min(H,y+3);x0=max(0,x-2);x1=min(W,x+3)
        nbr_labs=set(int(v) for v in np.unique(labels[y0:y1,x0:x1]) if int(v)>0)
        compatible=True
        for nl in nbr_labs:
            if nl==lab:
                continue
            ncid=label_to_id[nl]
            if ncid not in adj.get(cid,set()):
                compatible=False
                break
        if not compatible:
            rejected_topology[y,x]=True
            continue

        projected[y,x]=lab
        accepted[y,x]=True

    stats={
        "observed_fraction_of_tissue":float(observed.sum()/max(tissue.sum(),1)),
        "candidate_unassigned_fraction_of_tissue":float(candidate.sum()/max(tissue.sum(),1)),
        "accepted_reconstruction_fraction_of_tissue":float(accepted.sum()/max(tissue.sum(),1)),
        "remaining_unassigned_fraction_of_tissue":float(
            ((tissue)&(projected==0)).sum()/max(tissue.sum(),1)
        ),
        "rejected_distance_fraction_of_candidate":float(
            rejected_distance.sum()/max(candidate.sum(),1)
        ),
        "rejected_topology_fraction_of_candidate":float(
            rejected_topology.sum()/max(candidate.sum(),1)
        ),
        "observed_pixels_preserved":bool(np.all(projected[observed]==labels[observed])),
    }
    return projected,labels,evidence,meta,stats


def labels_to_boundary_table(label_img, meta):
    """
    Convert reconstructed raster territories back to polygon boundaries.
    Uses skimage only if available; otherwise returns an empty table and keeps
    the raster artifact as the authoritative reconstruction.
    """
    try:
        from skimage import measure
    except Exception:
        return pd.DataFrame()

    xmin,ymin,xmax,ymax=meta["bounds"]
    scale=meta["scale"]
    label_to_id=meta["label_to_cell_id"]
    rows=[]

    for lab,cid in label_to_id.items():
        mask=label_img==int(lab)
        if not np.any(mask):
            continue
        contours=measure.find_contours(mask.astype(float),0.5)
        if not contours:
            continue
        cont=max(contours,key=len)
        for k,(yy,xx) in enumerate(cont):
            x=xmin+(xx-1)/scale
            y=ymin+(yy-1)/scale
            rows.append({
                "cell_id":cid,
                "vertex_order":k,
                "vertex_x":float(x),
                "vertex_y":float(y),
            })
    return pd.DataFrame(rows)
