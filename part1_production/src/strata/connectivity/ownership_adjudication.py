
from __future__ import annotations

# v0.5.14: replace the hard target>=donor overlap veto from v0.5.13 by a
# continuous ambiguity cost. A contested pixel is still admissible ONLY when
# both continuous Xenium polygons physically intersect that pixel, and the
# donor must remain connected after transfer. The geometric overlap ratio is
# now a cost, not an absolute veto.

from dataclasses import dataclass
import heapq
import math
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
from shapely.geometry import box

ST4=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)

@dataclass(frozen=True)
class OwnershipConfig:
    halo_pixels:int=4
    max_passes:int=6
    area_epsilon:float=1e-12
    contested_base_cost:float=2.0
    donor_area_weight:float=2.0

def component_count(mask):
    _,n=ndimage.label(np.asarray(mask,bool),structure=ST4)
    return int(n)

def audit_disconnected(labels):
    labels=np.asarray(labels,dtype=np.int32)
    maxlab=int(labels.max()) if labels.size else 0
    slices=ndimage.find_objects(labels,max_label=maxlab)
    bad=[]
    for lab in range(1,maxlab+1):
        sl=slices[lab-1] if lab-1<len(slices) else None
        if sl is None: continue
        if component_count(labels[sl]==lab)>1: bad.append(lab)
    return bad

def topology_gate(*,post_disconnected_labels,gateB_contact_recall,new_contact_burden):
    # Whole-specimen LCC is intentionally not an existence gate.
    return bool(
        int(post_disconnected_labels)==0
        and float(gateB_contact_recall)>=0.95
        and float(new_contact_burden)<=0.35
    )

def _pixel_box(meta,y,x):
    xmin,ymin,_,_=meta["bounds"]; scale=float(meta["scale"])
    x0=xmin+(x-1)/scale; y0=ymin+(y-1)/scale; h=1.0/scale
    return box(x0,y0,x0+h,y0+h)

def _coverage(poly,pix):
    if poly is None or poly.is_empty:return 0.0
    try:return float(poly.intersection(pix).area)
    except Exception:return 0.0

def _donor_safe_after_removal(work,donor,remove_global):
    yy,xx=np.nonzero(work==donor)
    if len(yy)==0:return False
    y0,y1=int(yy.min()),int(yy.max())+1
    x0,x1=int(xx.min()),int(xx.max())+1
    before=(work[y0:y1,x0:x1]==donor)
    nb=component_count(before)
    after=before.copy()
    for y,x in remove_global:
        if y0<=y<y1 and x0<=x<x1:
            after[y-y0,x-x0]=False
    if not after.any():return False
    return component_count(after)<=nb

def _path(work,target,poly,polygons_by_label,meta,sl,cfg,forbidden=None):
    forbidden=set() if forbidden is None else set(forbidden)
    yoff,xoff=sl[0].start,sl[1].start
    crop=work[sl]
    m=crop==target
    cc,n=ndimage.label(m,structure=ST4)
    if n<=1:return [],True
    sizes=np.bincount(cc.ravel(),minlength=n+1)
    anchor=int(np.argmax(sizes[1:])+1)
    source=cc==anchor; targets=(cc>0)&(cc!=anchor)
    H,W=crop.shape
    dist=np.full((H,W),np.inf)
    py=np.full((H,W),-1,np.int32); px=np.full((H,W),-1,np.int32)
    heap=[]
    sy,sx=np.nonzero(source)
    for y,x in zip(sy,sx):
        dist[y,x]=0.;heapq.heappush(heap,(0.,int(y),int(x)))
    cov={}

    def coverage(label,gy,gx,p):
        k=(int(label),int(gy),int(gx))
        if k not in cov: cov[k]=_coverage(p,_pixel_box(meta,gy,gx))
        return cov[k]

    def cost(y,x):
        gy,gx=y+yoff,x+xoff
        if (gy,gx) in forbidden:return math.inf,False
        owner=int(crop[y,x])
        if owner==target:return 0.,True
        ta=coverage(target,gy,gx,poly)
        if ta<=cfg.area_epsilon:return math.inf,False
        if owner==0:return 1.,True

        donor_poly=polygons_by_label.get(owner)
        if donor_poly is None:return math.inf,False
        da=coverage(owner,gy,gx,donor_poly)

        # Scientific condition: a foreign-owned pixel is contested only if the
        # donor polygon also has physical support there. If it does not, the
        # raster ownership is plainly inconsistent with the continuous polygons.
        if da<=cfg.area_epsilon:return 1.25,True

        # IMPORTANT v0.5.14 change:
        # Do not veto a pixel merely because donor overlap area is larger.
        # Pixel-level overlap magnitude cannot uniquely decide ownership when
        # that choice destroys cell topology. Instead penalize donor-favored
        # pixels continuously and let shortest-path + donor-connectivity
        # constraints choose the minimum-distortion topological correction.
        frac=da/max(ta+da,cfg.area_epsilon)
        return cfg.contested_base_cost+cfg.donor_area_weight*frac,True

    hit=None
    while heap:
        d,y,x=heapq.heappop(heap)
        if d!=dist[y,x]:continue
        if targets[y,x]:
            hit=(y,x);break
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            ny,nx=y+dy,x+dx
            if not (0<=ny<H and 0<=nx<W):continue
            c,ok=cost(ny,nx)
            if not ok:continue
            nd=d+c
            if nd<dist[ny,nx]:
                dist[ny,nx]=nd;py[ny,nx]=y;px[ny,nx]=x
                heapq.heappush(heap,(nd,ny,nx))
    if hit is None:return [],False
    out=[];y,x=hit
    while not source[y,x]:
        out.append((y+yoff,x+xoff))
        yy,xx=int(py[y,x]),int(px[y,x])
        if yy<0:return [],False
        y,x=yy,xx
    out.reverse()
    return out,True

def adjudicate_ownership(labels,polygons_by_label,meta,cfg=OwnershipConfig()):
    work=np.asarray(labels,dtype=np.int32).copy()
    initial=audit_disconnected(work)
    records=[];total=bg_total=contested_total=0;touched=set()

    for pass_index in range(int(cfg.max_passes)):
        bad=audit_disconnected(work)
        if not bad:break
        changed=0
        for lab in sorted(bad):
            poly=polygons_by_label.get(int(lab))
            if poly is None or poly.is_empty:
                records.append({"pass":pass_index,"label":int(lab),"status":"NO_POLYGON"})
                continue
            yy,xx=np.nonzero(work==lab)
            if len(yy)==0:continue
            h=int(cfg.halo_pixels)
            sl=(slice(max(0,int(yy.min())-h),min(work.shape[0],int(yy.max())+h+1)),
                slice(max(0,int(xx.min())-h),min(work.shape[1],int(xx.max())+h+1)))
            before=component_count(work[sl]==lab)

            # If the minimum path would fragment a donor, forbid that path's
            # contested donor pixels and seek an alternative rather than
            # immediately giving up.
            forbidden=set()
            chosen=None
            block_count=0
            for attempt in range(8):
                path,ok=_path(work,lab,poly,polygons_by_label,meta,sl,cfg,forbidden)
                if not ok or not path:break
                donor_pixels={}
                for y,x in path:
                    owner=int(work[y,x])
                    if owner not in (0,lab):
                        donor_pixels.setdefault(owner,[]).append((y,x))
                unsafe=[]
                for donor,pix in donor_pixels.items():
                    if not _donor_safe_after_removal(work,donor,pix):
                        unsafe.extend(pix)
                if not unsafe:
                    chosen=(path,donor_pixels);break
                block_count+=1
                forbidden.update(unsafe)

            if chosen is None:
                records.append({
                    "pass":pass_index,"label":int(lab),
                    "status":"NO_SAFE_PATH",
                    "components_before":before,
                    "donor_blocked_alternatives":block_count
                })
                continue

            path,donor_pixels=chosen
            trial=work.copy();bg=ct=0;donors=set()
            for y,x in path:
                owner=int(trial[y,x])
                if owner==lab:continue
                if owner==0:bg+=1
                else:ct+=1;donors.add(owner)
                trial[y,x]=lab
            after=component_count(trial[sl]==lab)
            if after>=before:
                records.append({
                    "pass":pass_index,"label":int(lab),
                    "status":"NO_COMPONENT_REDUCTION",
                    "components_before":before,"components_after":after
                })
                continue
            work=trial;changed+=1;total+=bg+ct;bg_total+=bg;contested_total+=ct;touched.update(donors)
            records.append({
                "pass":pass_index,"label":int(lab),"status":"REPAIRED",
                "components_before":before,"components_after":after,
                "background_pixels_claimed":bg,
                "contested_pixels_reassigned":ct,
                "donor_count":len(donors),
                "donor_blocked_alternatives":block_count
            })
        if changed==0:break

    final=audit_disconnected(work)
    return work,{
        "initial_disconnected_labels":len(initial),
        "post_adjudication_disconnected_labels":len(final),
        "post_adjudication_examples":final[:20],
        "pixels_reassigned_total":int(total),
        "background_pixels_claimed":int(bg_total),
        "contested_pixels_reassigned":int(contested_total),
        "donor_labels_touched":int(len(touched)),
        "passes_completed":int(max([r["pass"] for r in records],default=-1)+1),
        "records":records
    }
