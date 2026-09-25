from __future__ import annotations
import heapq
import numpy as np
from scipy import ndimage
import shapely

ST4=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)


def _component_count(mask):
    _,n=ndimage.label(mask,structure=ST4)
    return int(n)


def remove_detached_inferred_fragments(labels, observed):
    """
    For each label, retain every connected component containing observed support.
    Completed-only components with no observed support are set to background.
    Observed pixels are never removed.
    """
    labels=np.asarray(labels,dtype=np.int32)
    observed=np.asarray(observed,dtype=np.int32)
    out=labels.copy()
    maxlab=int(labels.max())
    slices=ndimage.find_objects(labels,max_label=maxlab)
    changed_labels=0
    removed_pixels=0

    for lab in range(1,maxlab+1):
        sl=slices[lab-1] if lab-1<len(slices) else None
        if sl is None: continue
        m=(out[sl]==lab)
        cc,n=ndimage.label(m,structure=ST4)
        if n<=1: continue
        obs=(observed[sl]==lab)
        keep_ids=np.unique(cc[obs]); keep_ids=keep_ids[keep_ids>0]
        if len(keep_ids)==0:
            continue
        bad=m & ~np.isin(cc,keep_ids)
        if bad.any():
            changed_labels += 1
            removed_pixels += int(bad.sum())
            crop=out[sl]
            crop[bad]=0
            out[sl]=crop
    return out,changed_labels,removed_pixels


def _pixel_center(meta, yy, xx):
    xmin,ymin,_,_=meta["bounds"]
    scale=float(meta["scale"])
    px=xmin+(xx-1+0.5)/scale
    py=ymin+(yy-1+0.5)/scale
    return px,py


def _allowed_polygon_pixels(poly, meta, sl, pad_phys=None):
    """
    Candidate pixels whose centers lie in the measured polygon, or within half
    a raster diagonal of it. The tiny buffer handles center-sampling ambiguity.
    """
    ymin,ymax=sl[0].start,sl[0].stop
    xminp,xmaxp=sl[1].start,sl[1].stop
    yy,xx=np.mgrid[ymin:ymax,xminp:xmaxp]
    px,py=_pixel_center(meta,yy,xx)
    scale=float(meta["scale"])
    if pad_phys is None:
        pad_phys=0.72/scale  # ~half pixel diagonal
    qpoly=poly.buffer(pad_phys)
    allowed=shapely.contains_xy(qpoly,px.ravel(),py.ravel()).reshape(px.shape)
    return allowed


def _shortest_background_bridge(label_crop, allowed, blocked):
    """
    Multi-source Dijkstra/BFS on 4-neighbor pixels.

    Existing label pixels have zero cost, admissible background has unit cost,
    pixels owned by another cell are forbidden. Returns bridge pixels that join
    all current components when possible.
    """
    cc,n=ndimage.label(label_crop,structure=ST4)
    if n<=1:
        return np.zeros_like(label_crop,bool),True

    # Join components iteratively to the current anchor component.
    bridge=np.zeros_like(label_crop,bool)
    work=label_crop.copy()
    while True:
        cc,n=ndimage.label(work,structure=ST4)
        if n<=1:
            return bridge,True
        sizes=np.bincount(cc.ravel(),minlength=n+1)
        anchor=int(np.argmax(sizes[1:])+1)
        targets=(cc>0)&(cc!=anchor)
        source=(cc==anchor)

        H,W=work.shape
        dist=np.full((H,W),np.inf)
        prev_y=np.full((H,W),-1,np.int32)
        prev_x=np.full((H,W),-1,np.int32)
        heap=[]
        sy,sx=np.nonzero(source)
        for y,x in zip(sy,sx):
            dist[y,x]=0.0
            heapq.heappush(heap,(0.0,int(y),int(x)))

        hit=None
        while heap:
            d,y,x=heapq.heappop(heap)
            if d!=dist[y,x]: continue
            if targets[y,x]:
                hit=(y,x);break
            for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                ny,nx=y+dy,x+dx
                if ny<0 or ny>=H or nx<0 or nx>=W: continue
                if blocked[ny,nx]: continue
                if not (allowed[ny,nx] or work[ny,nx]): continue
                step=0.0 if work[ny,nx] else 1.0
                nd=d+step
                if nd<dist[ny,nx]:
                    dist[ny,nx]=nd
                    prev_y[ny,nx]=y;prev_x[ny,nx]=x
                    heapq.heappush(heap,(nd,ny,nx))
        if hit is None:
            return bridge,False

        y,x=hit
        while not source[y,x]:
            if not work[y,x]:
                bridge[y,x]=True
                work[y,x]=True
            py,px=prev_y[y,x],prev_x[y,x]
            if py<0: break
            y,x=int(py),int(px)


def connect_observed_polygon_raster(labels, observed, polygons, meta):
    """
    Repair only labels whose observed raster representation is disconnected.

    The bridge is constrained to the corresponding continuous measured polygon
    (plus <= half-pixel-diagonal tolerance) and may traverse background only.
    It may never overwrite another cell label.
    """
    out=np.asarray(labels,dtype=np.int32).copy()
    observed=np.asarray(observed,dtype=np.int32)
    id_to_lab=meta["cell_to_label"]
    repaired=0; added=0; failed=[]

    for cid,poly in polygons.items():
        lab=id_to_lab.get(str(cid))
        if lab is None: continue
        ys,xs=np.nonzero(observed==lab)
        if len(ys)==0: continue

        y0=max(0,int(ys.min())-4);y1=min(out.shape[0],int(ys.max())+5)
        x0=max(0,int(xs.min())-4);x1=min(out.shape[1],int(xs.max())+5)
        sl=(slice(y0,y1),slice(x0,x1))
        obs_crop=(observed[sl]==lab)
        if _component_count(obs_crop)<=1:
            continue

        label_crop=(out[sl]==lab)
        allowed=_allowed_polygon_pixels(poly,meta,sl)
        blocked=(out[sl]!=0)&(out[sl]!=lab)
        bridge,ok=_shortest_background_bridge(label_crop,allowed,blocked)
        if not ok:
            failed.append(str(cid))
            continue
        if bridge.any():
            crop=out[sl]
            crop[bridge]=lab
            out[sl]=crop
            repaired += 1
            added += int(bridge.sum())

    return out,{
        "observed_raster_split_labels_repaired":repaired,
        "bridge_pixels_added":added,
        "bridge_failures":len(failed),
        "bridge_failure_examples":failed[:20],
    }


def audit_disconnected(labels):
    labels=np.asarray(labels,dtype=np.int32)
    maxlab=int(labels.max())
    slices=ndimage.find_objects(labels,max_label=maxlab)
    bad=[]
    for lab in range(1,maxlab+1):
        sl=slices[lab-1] if lab-1<len(slices) else None
        if sl is None: continue
        if _component_count(labels[sl]==lab)>1:
            bad.append(lab)
    return bad
