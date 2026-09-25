from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
import numpy as np
import pandas as pd
from scipy import ndimage


@dataclass(frozen=True)
class JunctionRecoveryConfig:
    local_pad: int = 5
    min_angular_separation_deg: float = 12.0
    max_angular_sector_deg: float = 210.0
    require_exactly_three_cells: bool = True


def _local_pair_contacts(mask: np.ndarray, cells: tuple[int,int,int], sl):
    """
    Find measured 4-neighbour contacts between each pair of candidate cells
    inside a local crop. Returns pair -> contact midpoint coordinates in global
    x,y coordinates.
    """
    y0,y1,x0,x1=sl
    crop=mask[y0:y1,x0:x1]
    wanted={tuple(sorted(p)) for p in (
        (cells[0],cells[1]),(cells[0],cells[2]),(cells[1],cells[2])
    )}
    pts={p:[] for p in wanted}

    # horizontal label transitions
    a=crop[:,:-1]; b=crop[:,1:]
    q=(a!=b)&(a>0)&(b>0)
    ys,xs=np.nonzero(q)
    for y,x in zip(ys,xs):
        p=tuple(sorted((int(a[y,x]),int(b[y,x]))))
        if p in pts:
            pts[p].append((x0+x+1.0, y0+y+0.5))

    # vertical label transitions
    a=crop[:-1,:]; b=crop[1:,:]
    q=(a!=b)&(a>0)&(b>0)
    ys,xs=np.nonzero(q)
    for y,x in zip(ys,xs):
        p=tuple(sorted((int(a[y,x]),int(b[y,x]))))
        if p in pts:
            pts[p].append((x0+x+0.5, y0+y+1.0))
    return pts


def _angular_geometry_ok(center_xy, pair_pts, cfg:JunctionRecoveryConfig):
    """
    Require the three measured pairwise contact directions around the micro-gap
    to be geometrically non-degenerate.
    """
    cx,cy=center_xy
    angles=[]
    for pts in pair_pts.values():
        if not pts:
            return False, {}
        arr=np.asarray(pts,float)
        d2=(arr[:,0]-cx)**2+(arr[:,1]-cy)**2
        p=arr[int(np.argmin(d2))]
        angles.append(math.atan2(p[1]-cy,p[0]-cx)%(2*math.pi))
    angles=np.sort(np.asarray(angles))
    gaps=np.diff(np.r_[angles,angles[0]+2*math.pi])
    min_sep=float(np.min(gaps)*180/math.pi)
    max_sector=float(np.max(gaps)*180/math.pi)
    ok=(min_sep>=cfg.min_angular_separation_deg and
        max_sector<=cfg.max_angular_sector_deg)
    return ok, {
        "min_angular_separation_deg":min_sep,
        "max_angular_sector_deg":max_sector,
    }


def recover_microgap_junctions(mask:np.ndarray, bg_labels:np.ndarray,
                               persistence:pd.DataFrame,
                               interfaces:pd.DataFrame,
                               junctions:pd.DataFrame,
                               cfg:JunctionRecoveryConfig):
    """
    Conservative mechanics-only junction recovery.

    A junction is recovered only when:
      * the background component is classified unstable_micro_gap;
      * exactly three cells touch its one-pixel neighbourhood;
      * all three pairwise cell-cell interfaces exist in the measured interface
        table;
      * all three pairwise contacts are observed locally near the gap;
      * their directions around the gap are non-degenerate;
      * an equivalent junction does not already exist.

    Therefore recovery adds force-balance constraints without introducing new
    interface/tension variables.
    """
    unstable=set(
        persistence.loc[
            persistence.mechanical_boundary_class=="unstable_micro_gap",
            "background_component"
        ].astype(int)
    )
    pair_to_e={}
    for r in interfaces.itertuples():
        if r.kind=="cell_cell":
            pair_to_e[(min(int(r.cell_i),int(r.cell_j)),
                       max(int(r.cell_i),int(r.cell_j)))]=int(r.interface_id)

    existing=set()
    for r in junctions.itertuples():
        inc=tuple(sorted(int(e) for e in r.incident_interfaces))
        if len(inc)>=3:
            existing.add(inc)

    slices=ndimage.find_objects(bg_labels,max_label=int(bg_labels.max()))
    new_rows=[]
    audit=Counter()
    detail=[]

    for bc in sorted(unstable):
        sl0=slices[bc-1] if bc-1<len(slices) else None
        if sl0 is None:
            audit["missing_component"]+=1
            continue
        ys,xs=np.nonzero(bg_labels[sl0]==bc)
        if len(ys)==0:
            audit["missing_component"]+=1
            continue
        gy=ys+sl0[0].start; gx=xs+sl0[1].start
        cx=float(gx.mean()); cy=float(gy.mean())

        # One-pixel ring around this component.
        local=np.zeros(mask.shape,bool)
        local[gy,gx]=True
        ring=ndimage.binary_dilation(local,structure=np.ones((3,3),bool)) & ~local
        cells=sorted(set(int(x) for x in np.unique(mask[ring]) if x>0))
        if cfg.require_exactly_three_cells and len(cells)!=3:
            audit["reject_not_exactly_three_cells"]+=1
            continue
        if len(cells)!=3:
            audit["reject_cell_count"]+=1
            continue
        cells=tuple(cells)
        audit["exactly_three_cells"]+=1

        pairs=[
            tuple(sorted((cells[0],cells[1]))),
            tuple(sorted((cells[0],cells[2]))),
            tuple(sorted((cells[1],cells[2]))),
        ]
        if not all(p in pair_to_e for p in pairs):
            audit["reject_missing_pairwise_interface"]+=1
            continue
        audit["all_pairwise_interfaces_exist"]+=1

        y0=max(0,int(gy.min())-cfg.local_pad)
        y1=min(mask.shape[0],int(gy.max())+cfg.local_pad+1)
        x0=max(0,int(gx.min())-cfg.local_pad)
        x1=min(mask.shape[1],int(gx.max())+cfg.local_pad+1)
        pair_pts=_local_pair_contacts(mask,cells,(y0,y1,x0,x1))
        if not all(len(pair_pts[p])>0 for p in pairs):
            audit["reject_pairwise_contacts_not_local"]+=1
            continue
        audit["local_pairwise_contacts_exist"]+=1

        geom_ok,gmeta=_angular_geometry_ok((cx,cy),pair_pts,cfg)
        if not geom_ok:
            audit["reject_degenerate_contact_geometry"]+=1
            continue

        inc=tuple(sorted(pair_to_e[p] for p in pairs))
        if inc in existing:
            audit["reject_equivalent_junction_exists"]+=1
            continue

        jid=int(junctions.junction_id.max()+1+len(new_rows)) if len(junctions) else len(new_rows)
        new_rows.append({
            "junction_id":jid,
            "x":cx,"y":cy,
            "incident_interfaces":list(inc),
            "n_regions":3,
            "source":"recovered_unstable_microgap",
            "background_component":int(bc),
            "cell_a":int(cells[0]),"cell_b":int(cells[1]),"cell_c":int(cells[2]),
            **gmeta,
        })
        existing.add(inc)
        audit["recovered"]+=1
        detail.append({
            "background_component":int(bc),
            "junction_id":jid,
            "cell_a":cells[0],"cell_b":cells[1],"cell_c":cells[2],
            "interface_1":inc[0],"interface_2":inc[1],"interface_3":inc[2],
            **gmeta,
        })

    Jnew=pd.DataFrame(new_rows)
    if len(Jnew):
        Jout=pd.concat([junctions,Jnew],ignore_index=True,sort=False)
    else:
        Jout=junctions.copy()
    return Jout,pd.DataFrame(detail),dict(audit)


def patch_recovered_junction_count(cells, recovered_detail, interfaces):
    """
    Count recovered junctions whose three cells all lie in a patch.
    """
    cset=set(int(x) for x in cells)
    if len(recovered_detail)==0:
        return 0
    q=recovered_detail[
        recovered_detail.cell_a.astype(int).isin(cset) &
        recovered_detail.cell_b.astype(int).isin(cset) &
        recovered_detail.cell_c.astype(int).isin(cset)
    ]
    return int(len(q))
