from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.optimize import lsq_linear
from scipy.sparse.csgraph import structural_rank


@dataclass(frozen=True)
class SolverConfig:
    min_curvature_confidence: float = 0.15
    min_interface_length: float = 3.0
    junction_weight: float = 1.0
    curvature_weight: float = 1.0
    pressure_gauge_weight: float = 10.0
    tension_gauge_weight: float = 10.0
    max_iter: int = 300


def build_patch_system(cells, interfaces, junctions, cell_centroids, cfg: SolverConfig):
    cells=set(int(x) for x in cells)
    E=interfaces[
        interfaces.cell_i.astype(int).isin(cells) &
        ((interfaces.kind=="cell_boundary") | interfaces.cell_j.fillna(-1).astype(int).isin(cells))
    ].copy()

    eids=list(E.interface_id.astype(int))
    eidx={e:i for i,e in enumerate(eids)}
    cids=sorted(cells)
    cidx={c:len(eids)+i for i,c in enumerate(cids)}
    bcs=sorted(set(int(x) for x in E.loc[E.kind=="cell_boundary","background_component"].dropna()))
    bidx={b:len(eids)+len(cids)+i for i,b in enumerate(bcs)}
    nvar=len(eids)+len(cids)+len(bcs)

    rows=[]; cols=[]; vals=[]; rhs=[]; rowmeta=[]

    def add(coeffs, y, meta, w=1.0):
        r=len(rhs)
        for j,v in coeffs.items():
            rows.append(r); cols.append(j); vals.append(float(w*v))
        rhs.append(float(w*y)); rowmeta.append(meta)

    # Young-Laplace rows.
    for r in E.itertuples():
        if r.length < cfg.min_interface_length:
            continue
        if r.curvature_confidence < cfg.min_curvature_confidence:
            continue
        co={cidx[int(r.cell_i)]:1.0, eidx[int(r.interface_id)]:-float(r.curvature)}
        if r.kind=="cell_cell":
            co[cidx[int(r.cell_j)]]=-1.0
        else:
            co[bidx[int(r.background_component)]]=-1.0
        w=cfg.curvature_weight*max(float(r.curvature_confidence),0.05)
        add(co,0.0,("young_laplace",int(r.interface_id)),w)

    # Junction force-balance rows. Direction is from junction to interface centroid.
    J=junctions.copy()
    e_lookup=E.set_index("interface_id")
    for j in J.itertuples():
        inc=[int(e) for e in j.incident_interfaces if int(e) in eidx]
        if len(inc)<3:
            continue
        cx={}; cy={}
        for e in inc:
            er=e_lookup.loc[e]
            dx=float(er.centroid_x)-float(j.x)
            dy=float(er.centroid_y)-float(j.y)
            n=(dx*dx+dy*dy)**0.5
            if n<1e-12:
                continue
            cx[eidx[e]]=dx/n
            cy[eidx[e]]=dy/n
        if len(cx)>=3:
            add(cx,0.0,("junction_x",int(j.junction_id)),cfg.junction_weight)
            add(cy,0.0,("junction_y",int(j.junction_id)),cfg.junction_weight)

    # Explicit gauges.
    if eids:
        add({eidx[e]:1.0/len(eids) for e in eids},1.0,("tension_gauge",-1),cfg.tension_gauge_weight)
    if cids:
        add({cidx[c]:1.0/len(cids) for c in cids},0.0,("pressure_gauge",-1),cfg.pressure_gauge_weight)

    A=coo_matrix((vals,(rows,cols)),shape=(len(rhs),nvar)).tocsr()
    b=np.asarray(rhs,float)
    return A,b,{"eidx":eidx,"cidx":cidx,"bidx":bidx,"rows":rowmeta,"E":E}


def identifiability(A: csr_matrix):
    sr=int(structural_rank(A))
    nvar=int(A.shape[1])
    rank_frac=float(sr/max(nvar,1))
    # Approximate numerical conditioning from a dense sketch when modest.
    numerical_rank=None
    cond=None
    if min(A.shape) <= 1200:
        D=A.toarray()
        s=np.linalg.svd(D,compute_uv=False)
        if len(s):
            tol=max(D.shape)*np.finfo(float).eps*s[0]
            numerical_rank=int((s>tol).sum())
            nz=s[s>tol]
            cond=float(nz[0]/nz[-1]) if len(nz)>1 else 1.0
    return {
        "n_rows":int(A.shape[0]),
        "n_variables":nvar,
        "structural_rank":sr,
        "structural_rank_fraction":rank_frac,
        "numerical_rank":numerical_rank,
        "condition_estimate":cond,
        "structurally_identifiable":bool(sr>=nvar),
    }


def solve_patch(cells, interfaces, junctions, cell_centroids, cfg: SolverConfig):
    A,b,meta=build_patch_system(cells,interfaces,junctions,cell_centroids,cfg)
    ident=identifiability(A)
    if A.shape[0]==0 or A.shape[1]==0:
        return {"status":"HOLD","identifiability":ident}

    nE=len(meta["eidx"])
    lb=np.full(A.shape[1],-np.inf)
    ub=np.full(A.shape[1], np.inf)
    lb[:nE]=0.0

    res=lsq_linear(A,b,bounds=(lb,ub),method="trf",max_iter=cfg.max_iter,lsmr_tol="auto")
    x=res.x
    pred=A@x
    residual=pred-b
    scale=max(float(np.median(np.abs(b))) if len(b) else 1.0,1.0)
    rms=float(np.sqrt(np.mean(residual**2))/scale)
    q95=float(np.quantile(np.abs(residual),.95)/scale) if len(residual) else 0.0

    tensions={e:float(x[i]) for e,i in meta["eidx"].items()}
    pressures={c:float(x[i]) for c,i in meta["cidx"].items()}
    boundary_pressures={bc:float(x[i]) for bc,i in meta["bidx"].items()}

    neg_frac=float(np.mean(np.array(list(tensions.values()))<0)) if tensions else 0.0
    return {
        "status":"PASS" if res.success else "HOLD",
        "success":bool(res.success),
        "message":str(res.message),
        "cost":float(res.cost),
        "n_iterations":int(getattr(res,"nit",0) or 0),
        "residual_rms":rms,
        "residual_q95":q95,
        "identifiability":ident,
        "negative_tension_fraction":neg_frac,
        "tensions":tensions,
        "pressures":pressures,
        "boundary_pressures":boundary_pressures,
    }
