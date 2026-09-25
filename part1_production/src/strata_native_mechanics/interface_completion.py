from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import math
import numpy as np
import pandas as pd
from scipy import ndimage

@dataclass(frozen=True)
class InterfaceCompletionConfig:
    local_pad: int = 5
    min_virtual_contact_edges: int = 2
    max_gap_pixels: int = 64
    max_tie_fraction: float = 0.40
    min_angular_separation_deg: float = 8.0
    max_angular_sector_deg: float = 225.0

def _pairset(cells):
    a,b,c=sorted(int(x) for x in cells)
    return {(a,b),(a,c),(b,c)}

def _measured_pair_map(E):
    d={}
    for r in E.itertuples():
        if r.kind=="cell_cell":
            d[(min(int(r.cell_i),int(r.cell_j)),
               max(int(r.cell_i),int(r.cell_j)))]=int(r.interface_id)
    return d

def _nearest_cell_fill(mask, gap_y, gap_x, cells, sl):
    """
    Mechanics-only Voronoi assignment of a single micro-gap to its three
    surrounding measured cells. Returns the local completed label image and
    a tie diagnostic. The original mask is never modified.
    """
    y0,y1,x0,x1=sl
    crop=mask[y0:y1,x0:x1].copy()
    gy=gap_y-y0; gx=gap_x-x0
    dists=[]
    for c in cells:
        # EDT gives distance to nearest zero; make target cell zero.
        d=ndimage.distance_transform_edt(crop!=int(c))
        dists.append(d[gy,gx])
    D=np.vstack(dists).T
    order=np.sort(D,axis=1)
    ties=np.isclose(order[:,0],order[:,1],rtol=0,atol=1e-9)
    winner=np.argmin(D,axis=1)
    for k,c in enumerate(cells):
        q=(winner==k)
        crop[gy[q],gx[q]]=int(c)
    return crop,float(np.mean(ties)) if len(ties) else 0.0

def _contacts(crop,global_x0,global_y0,allowed_cells):
    allowed=set(int(x) for x in allowed_cells)
    pts={p:[] for p in _pairset(allowed)}
    a=crop[:,:-1];b=crop[:,1:]
    q=(a!=b)&(a>0)&(b>0)
    ys,xs=np.nonzero(q)
    for y,x in zip(ys,xs):
        p=tuple(sorted((int(a[y,x]),int(b[y,x]))))
        if p in pts:
            pts[p].append((global_x0+x+1.0,global_y0+y+0.5))
    a=crop[:-1,:];b=crop[1:,:]
    q=(a!=b)&(a>0)&(b>0)
    ys,xs=np.nonzero(q)
    for y,x in zip(ys,xs):
        p=tuple(sorted((int(a[y,x]),int(b[y,x]))))
        if p in pts:
            pts[p].append((global_x0+x+0.5,global_y0+y+1.0))
    return pts

def _pca_tangent(points):
    P=np.asarray(points,float)
    cen=P.mean(axis=0)
    if len(P)<2:
        return cen,np.array([1.0,0.0])
    X=P-cen
    C=X.T@X
    vals,vecs=np.linalg.eigh(C)
    t=vecs[:,int(np.argmax(vals))]
    return cen,t/max(np.linalg.norm(t),1e-12)

def _angular_ok(center, interface_ids, Eall, virtual_row, cfg):
    cx,cy=center
    lookup={int(r.interface_id):r for r in Eall.itertuples()}
    dirs=[]
    for eid in interface_ids:
        if eid==int(virtual_row["interface_id"]):
            x=float(virtual_row["centroid_x"]);y=float(virtual_row["centroid_y"])
        else:
            r=lookup[eid];x=float(r.centroid_x);y=float(r.centroid_y)
        dx=x-cx;dy=y-cy
        n=(dx*dx+dy*dy)**.5
        if n<1e-9:
            return False,{}
        dirs.append(math.atan2(dy,dx)%(2*math.pi))
    ang=np.sort(np.asarray(dirs))
    gaps=np.diff(np.r_[ang,ang[0]+2*math.pi])
    mn=float(gaps.min()*180/math.pi);mx=float(gaps.max()*180/math.pi)
    return (mn>=cfg.min_angular_separation_deg and mx<=cfg.max_angular_sector_deg),{
        "min_angular_separation_deg":mn,
        "max_angular_sector_deg":mx,
    }

def complete_interfaces_and_junctions(mask,bg_labels,persistence,E,J,cfg):
    """
    Counterfactual mechanics-only completion.

    Only unstable micro-gaps with exactly three surrounding cells and exactly
    ONE missing measured pairwise contact are eligible. The gap is assigned to
    surrounding cells by nearest-cell Voronoi completion. The missing pair must
    then acquire >= min_virtual_contact_edges locally.

    The virtual interface has curvature_confidence=0 and curvature=0, therefore
    it contributes no Young-Laplace row under the native solver defaults.
    It exists only so a uniquely implied force-balance junction can be audited.
    """
    unstable=set(persistence.loc[
        persistence.mechanical_boundary_class=="unstable_micro_gap",
        "background_component"].astype(int))
    pairmap=_measured_pair_map(E)
    slices=ndimage.find_objects(bg_labels,max_label=int(bg_labels.max()))
    Eout=E.copy()
    Jout=J.copy()
    detail=[];audit=Counter()
    next_e=int(E.interface_id.max()+1) if len(E) else 0
    next_j=int(J.junction_id.max()+1) if len(J) else 0
    existing_j={tuple(sorted(int(e) for e in r.incident_interfaces))
                for r in J.itertuples() if len(r.incident_interfaces)>=3}

    for bc in sorted(unstable):
        sl0=slices[bc-1] if bc-1<len(slices) else None
        if sl0 is None: continue
        gy0,gx0=np.nonzero(bg_labels[sl0]==bc)
        if len(gy0)==0 or len(gy0)>cfg.max_gap_pixels:
            audit["reject_size"]+=1;continue
        gy=gy0+sl0[0].start;gx=gx0+sl0[1].start
        comp=np.zeros(mask.shape,bool);comp[gy,gx]=True
        ring=ndimage.binary_dilation(comp,np.ones((3,3),bool))&~comp
        cells=sorted(int(x) for x in np.unique(mask[ring]) if x>0)
        if len(cells)!=3:
            audit["reject_not_three_cells"]+=1;continue
        audit["three_cells"]+=1
        allpairs=_pairset(cells)
        measured={p for p in allpairs if p in pairmap}
        missing=sorted(allpairs-measured)
        if len(missing)!=1:
            audit["reject_not_exactly_one_missing_pair"]+=1;continue
        audit["one_missing_pair"]+=1

        y0=max(0,int(gy.min())-cfg.local_pad);y1=min(mask.shape[0],int(gy.max())+cfg.local_pad+1)
        x0=max(0,int(gx.min())-cfg.local_pad);x1=min(mask.shape[1],int(gx.max())+cfg.local_pad+1)
        completed,tief=_nearest_cell_fill(mask,gy,gx,cells,(y0,y1,x0,x1))
        if tief>cfg.max_tie_fraction:
            audit["reject_tie_fraction"]+=1;continue

        pts=_contacts(completed,x0,y0,cells)
        mp=missing[0]
        if len(pts.get(mp,[]))<cfg.min_virtual_contact_edges:
            audit["reject_no_supported_virtual_contact"]+=1;continue
        audit["supported_virtual_contact"]+=1

        cen,t=_pca_tangent(pts[mp])
        vrow={
            "interface_id":next_e,
            "kind":"cell_cell",
            "cell_i":int(mp[0]),"cell_j":int(mp[1]),
            "background_component":None,
            "length":float(len(pts[mp])),
            "centroid_x":float(cen[0]),"centroid_y":float(cen[1]),
            "tangent_x":float(t[0]),"tangent_y":float(t[1]),
            "curvature":0.0,
            "curvature_confidence":0.0,
            "n_interface_pixels":int(len(pts[mp])),
            "source":"virtual_microgap_completion",
        }
        inc=tuple(sorted([pairmap[p] if p in pairmap else next_e for p in sorted(allpairs)]))
        if inc in existing_j:
            audit["reject_existing_equivalent_junction"]+=1;continue
        center=(float(gx.mean()),float(gy.mean()))
        ok,gmeta=_angular_ok(center,inc,Eout,vrow,cfg)
        if not ok:
            audit["reject_degenerate_geometry"]+=1;continue

        Eout=pd.concat([Eout,pd.DataFrame([vrow])],ignore_index=True,sort=False)
        jrow={
            "junction_id":next_j,
            "x":center[0],"y":center[1],
            "incident_interfaces":list(inc),
            "n_regions":3,
            "source":"virtual_microgap_completion",
            "background_component":int(bc),
        }
        Jout=pd.concat([Jout,pd.DataFrame([jrow])],ignore_index=True,sort=False)
        pairmap[mp]=next_e;existing_j.add(inc)
        detail.append({
            "background_component":int(bc),
            "virtual_interface_id":next_e,
            "recovered_junction_id":next_j,
            "cell_i":int(mp[0]),"cell_j":int(mp[1]),
            "third_cell":int(next(c for c in cells if c not in mp)),
            "virtual_contact_edges":int(len(pts[mp])),
            "tie_fraction":float(tief),
            **gmeta,
        })
        next_e+=1;next_j+=1;audit["accepted"]+=1

    return Eout,Jout,pd.DataFrame(detail),dict(audit)
