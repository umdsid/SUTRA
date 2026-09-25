from __future__ import annotations
import numpy as np
from scipy import ndimage

ST4=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
ST8=np.ones((3,3),dtype=np.uint8)

def ncomp(mask,st):
    _,n=ndimage.label(mask,structure=st)
    return int(n)

def foreign_neighbor_count(arr,y,x,lab):
    y0=max(0,y-1);y1=min(arr.shape[0],y+2)
    x0=max(0,x-1);x1=min(arr.shape[1],x+2)
    vals=np.unique(arr[y0:y1,x0:x1])
    return int(np.sum((vals!=0)&(vals!=lab)))

def bridge_diagonal_label(arr,lab):
    """
    Add background pixels only. Never overwrite another label.

    Repeatedly identify diagonal contacts that connect distinct 4-connected
    components of the same 8-connected label. For each such diagonal contact,
    fill one of the two orthogonal corner pixels if background. The candidate
    with fewer adjacent foreign labels is preferred deterministically.
    """
    out=arr
    ys,xs=np.nonzero(out==lab)
    if len(ys)==0:
        return 0,False

    y0=max(0,int(ys.min())-2); y1=min(out.shape[0],int(ys.max())+3)
    x0=max(0,int(xs.min())-2); x1=min(out.shape[1],int(xs.max())+3)
    sl=(slice(y0,y1),slice(x0,x1))
    crop=out[sl].copy()
    added=0

    # Only meant for 8-connected / 4-disconnected labels.
    if ncomp(crop==lab,ST8)!=1:
        return 0,False

    max_iter=max(16,int((crop==lab).sum()))
    for _ in range(max_iter):
        cc,n=ndimage.label(crop==lab,structure=ST4)
        if n<=1:
            out[sl]=crop
            return added,True

        proposals=[]
        # Search diagonal same-label pairs belonging to different 4-components.
        ly,lx=np.nonzero(crop==lab)
        for y,x in zip(ly,lx):
            c0=cc[y,x]
            for dy,dx in ((-1,-1),(-1,1),(1,-1),(1,1)):
                yy,xx=y+dy,x+dx
                if yy<0 or yy>=crop.shape[0] or xx<0 or xx>=crop.shape[1]:
                    continue
                if crop[yy,xx]!=lab or cc[yy,xx]==c0:
                    continue
                # Two orthogonal corner candidates.
                cands=[(y,xx),(yy,x)]
                valid=[]
                for cy,cx in cands:
                    if crop[cy,cx]==0:
                        score=foreign_neighbor_count(crop,cy,cx,lab)
                        valid.append((score,cy,cx))
                if valid:
                    proposals.extend(valid)

        if not proposals:
            return added,False

        proposals.sort(key=lambda t:(t[0],t[1],t[2]))
        _,cy,cx=proposals[0]
        crop[cy,cx]=lab
        added+=1

    return added,False

def repair_diagonal_only_labels(labels):
    out=np.asarray(labels,dtype=np.int32).copy()
    maxlab=int(out.max())
    slices=ndimage.find_objects(out,max_label=maxlab)
    four_bad=[]; eight_bad=[]

    for lab in range(1,maxlab+1):
        sl=slices[lab-1] if lab-1<len(slices) else None
        if sl is None: continue
        m=out[sl]==lab
        if ncomp(m,ST4)>1:
            four_bad.append(lab)
            if ncomp(m,ST8)>1:
                eight_bad.append(lab)

    diagonal_only=sorted(set(four_bad)-set(eight_bad))
    added_by_label={}
    failed=[]
    for lab in diagonal_only:
        n,ok=bridge_diagonal_label(out,lab)
        added_by_label[int(lab)]=int(n)
        if not ok:
            failed.append(int(lab))

    # Post audit.
    post4=[]; post8=[]
    slices=ndimage.find_objects(out,max_label=maxlab)
    for lab in range(1,maxlab+1):
        sl=slices[lab-1] if lab-1<len(slices) else None
        if sl is None: continue
        m=out[sl]==lab
        if ncomp(m,ST4)>1: post4.append(lab)
        if ncomp(m,ST8)>1: post8.append(lab)

    return out,{
        "pre_disconnected_4":len(four_bad),
        "pre_disconnected_8":len(eight_bad),
        "diagonal_only_labels":len(diagonal_only),
        "diagonal_bridge_failures":len(failed),
        "diagonal_bridge_failure_examples":failed[:20],
        "bridge_pixels_added":int(sum(added_by_label.values())),
        "max_bridge_pixels_per_label":int(max(added_by_label.values(),default=0)),
        "post_disconnected_4":len(post4),
        "post_disconnected_8":len(post8),
        "post_disconnected_4_examples":post4[:30],
        "post_disconnected_8_examples":post8[:30],
    }
