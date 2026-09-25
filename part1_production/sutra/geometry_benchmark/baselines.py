from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import ndimage
from shapely.geometry import Polygon


def polygons_from_boundaries(df):
    xcol="vertex_x" if "vertex_x" in df.columns else "x"
    ycol="vertex_y" if "vertex_y" in df.columns else "y"
    order=None
    for c in ("vertex_order","vertex_index","vertex_id"):
        if c in df.columns:
            order=c; break
    out={}
    for cid,g in df.groupby("cell_id",sort=False):
        if order: g=g.sort_values(order)
        xy=g[[xcol,ycol]].to_numpy(float)
        xy=xy[np.isfinite(xy).all(axis=1)]
        if len(xy)<3: continue
        p=Polygon(xy)
        if not p.is_valid: p=p.buffer(0)
        if not p.is_empty: out[str(cid)]=p
    return out


def make_raster(polys, cells, max_pixels=8_000_000):
    b=np.array([p.bounds for p in polys.values()],float)
    xmin,ymin=float(b[:,0].min()),float(b[:,1].min())
    xmax,ymax=float(b[:,2].max()),float(b[:,3].max())
    w=max(xmax-xmin,1e-9);h=max(ymax-ymin,1e-9)
    scale=min(max(math.sqrt(max_pixels/(w*h)),0.20),3.0)
    W=max(8,int(math.ceil(w*scale))+3)
    H=max(8,int(math.ceil(h*scale))+3)
    labels=np.zeros((H,W),np.int32)
    ids=sorted(polys.keys())
    tolab={cid:i+1 for i,cid in enumerate(ids)}
    toid={i+1:cid for i,cid in enumerate(ids)}

    import shapely
    for cid,p in polys.items():
        bx0=max(0,int(math.floor((p.bounds[0]-xmin)*scale))+1)
        bx1=min(W,int(math.ceil((p.bounds[2]-xmin)*scale))+2)
        by0=max(0,int(math.floor((p.bounds[1]-ymin)*scale))+1)
        by1=min(H,int(math.ceil((p.bounds[3]-ymin)*scale))+2)
        if bx1<=bx0 or by1<=by0: continue
        yy,xx=np.mgrid[by0:by1,bx0:bx1]
        px=xmin+(xx-1+0.5)/scale
        py=ymin+(yy-1+0.5)/scale
        inside=shapely.contains_xy(p,px.ravel(),py.ravel()).reshape(px.shape)
        labels[by0:by1,bx0:bx1][inside]=tolab[cid]

    return labels,{
        "bounds":[xmin,ymin,xmax,ymax],"scale":scale,"shape":[H,W],
        "cell_to_label":tolab,"label_to_cell":toid,
    }


def morphology_like_envelope(observed):
    occ=observed>0
    closed=ndimage.binary_closing(occ,iterations=2)
    env=ndimage.binary_fill_holes(closed)
    return env


def constrained_nearest_completion(observed, meta, gateb_edges, max_distance=8.0):
    env=morphology_like_envelope(observed)
    occ=observed>0
    _,inds=ndimage.distance_transform_edt(~occ,return_indices=True)
    nearest=observed[tuple(inds)]
    dphys=ndimage.distance_transform_edt(~occ)/max(meta["scale"],1e-12)

    adj={}
    for a,b in gateb_edges:
        adj.setdefault(str(a),set()).add(str(b))
        adj.setdefault(str(b),set()).add(str(a))

    toid=meta["label_to_cell"]
    out=observed.copy()
    cand=env & ~occ
    accepted=np.zeros_like(cand,bool)

    ys,xs=np.nonzero(cand & (dphys<=max_distance))
    for y,x in zip(ys,xs):
        lab=int(nearest[y,x])
        if lab<=0: continue
        cid=toid[lab]
        y0=max(0,y-1);y1=min(out.shape[0],y+2)
        x0=max(0,x-1);x1=min(out.shape[1],x+2)
        nbr=set(int(v) for v in np.unique(observed[y0:y1,x0:x1]) if int(v)>0)
        ok=True
        for nl in nbr:
            if nl==lab: continue
            nid=toid[nl]
            if nid not in adj.get(cid,set()):
                ok=False;break
        if ok:
            out[y,x]=lab;accepted[y,x]=True

    return out,{
        "method":"constrained_nearest",
        "candidate_fraction":float(cand.sum()/max(env.sum(),1)),
        "accepted_fraction":float(accepted.sum()/max(env.sum(),1)),
        "remaining_unassigned_fraction":float(((env)&(out==0)).sum()/max(env.sum(),1)),
        "observed_preserved":bool(np.all(out[occ]==observed[occ])),
    }


def transcript_density(transcripts, meta):
    xcol="x_location" if "x_location" in transcripts.columns else "x"
    ycol="y_location" if "y_location" in transcripts.columns else "y"
    xmin,ymin,xmax,ymax=meta["bounds"];scale=meta["scale"]
    H,W=meta["shape"]
    x=pd.to_numeric(transcripts[xcol],errors="coerce").to_numpy(float)
    y=pd.to_numeric(transcripts[ycol],errors="coerce").to_numpy(float)
    finite=np.isfinite(x)&np.isfinite(y)
    ix=np.zeros(len(x),int); iy=np.zeros(len(y),int)
    ix[finite]=np.floor((x[finite]-xmin)*scale+1).astype(int)
    iy[finite]=np.floor((y[finite]-ymin)*scale+1).astype(int)
    q=finite&(ix>=0)&(ix<W)&(iy>=0)&(iy<H)
    h=np.zeros((H,W),np.float32)
    np.add.at(h,(iy[q],ix[q]),1.0)
    return ndimage.gaussian_filter(h,1.2)


def transcript_watershed_completion(observed, density):
    """
    Transcript-supported marker watershed using SciPy only.

    High transcript density lowers the topographic cost. Observed cell
    territories are immutable markers. The benchmark envelope is the only
    admissible reconstruction domain.
    """
    env=morphology_like_envelope(observed)
    d=density.astype(np.float64)
    if np.max(d)>0:
        d=d/np.max(d)
    cost=np.clip(np.rint(255.0*(1.0-d)),0,255).astype(np.uint8)
    cost[~env]=255

    markers=observed.astype(np.int32,copy=True)
    out=ndimage.watershed_ift(cost,markers)
    out[~env]=0
    occ=observed>0
    out[occ]=observed[occ]

    return out,{
        "method":"transcript_watershed_scipy",
        "remaining_unassigned_fraction":float(((env)&(out==0)).sum()/max(env.sum(),1)),
        "observed_preserved":bool(np.all(out[occ]==observed[occ])),
    }


def adjacency_from_labels(labels):
    pairs=set()
    for a,b in ((labels[:,:-1],labels[:,1:]),(labels[:-1,:],labels[1:,:])):
        q=(a!=b)&(a>0)&(b>0)
        for x,y in zip(a[q],b[q]):
            pairs.add(tuple(sorted((int(x),int(y)))))
    return pairs


def score_labels(labels, observed, meta, gateb_edges, transcript_xy=None):
    toid=meta["label_to_cell"]
    pairs_lab=adjacency_from_labels(labels)
    pred={tuple(sorted((toid[a],toid[b]))) for a,b in pairs_lab if a in toid and b in toid}
    truth={tuple(sorted((str(a),str(b)))) for a,b in gateb_edges}
    tp=len(pred&truth)
    precision=tp/max(len(pred),1)
    recall=tp/max(len(truth),1)

    labs=np.array(sorted(toid.keys()),int)
    old=np.bincount(observed.ravel(),minlength=int(labs.max())+1)[labs]
    new=np.bincount(labels.ravel(),minlength=int(labs.max())+1)[labs]
    ratio=(new+1)/(old+1)
    med_abs_log=float(np.median(np.abs(np.log(ratio))))

    out={
        "contact_precision_vs_gateB":float(precision),
        "contact_recall_vs_gateB":float(recall),
        "contact_f1_vs_gateB":float(2*precision*recall/max(precision+recall,1e-15)),
        "new_contact_burden":float(len(pred-truth)/max(len(pred),1)),
        "lost_gateB_fraction":float(len(truth-pred)/max(len(truth),1)),
        "median_abs_log_area_ratio":med_abs_log,
        "n_contacts":len(pred),
    }

    if transcript_xy is not None and len(transcript_xy):
        xmin,ymin,xmax,ymax=meta["bounds"];scale=meta["scale"];H,W=meta["shape"]
        finite=np.isfinite(transcript_xy).all(axis=1)
        ix=np.zeros(len(transcript_xy),int); iy=np.zeros(len(transcript_xy),int)
        ix[finite]=np.floor((transcript_xy[finite,0]-xmin)*scale+1).astype(int)
        iy[finite]=np.floor((transcript_xy[finite,1]-ymin)*scale+1).astype(int)
        q=finite&(ix>=0)&(ix<W)&(iy>=0)&(iy<H)
        if q.any():
            out["transcript_capture_fraction"]=float(np.mean(labels[iy[q],ix[q]]>0))
    return out
