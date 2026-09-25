from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy import ndimage

from strata_native_mechanics.solver import SolverConfig, build_patch_system
from strata_native_mechanics.identifiability_v2 import rank_nullity


@dataclass(frozen=True)
class CorrectionConfig:
    persistence_radii: tuple = (1,2,3)
    unstable_area_max: int = 64
    unstable_survival_max: float = 0.10
    persistent_survival_min: float = 0.50
    curvature_confidence_min: float = 0.15
    curvature_abs_min: float = 1e-3
    min_junction_incidence_for_low_curvature: int = 2


def disk(radius:int):
    y,x=np.ogrid[-radius:radius+1,-radius:radius+1]
    return (x*x+y*y)<=radius*radius


def classify_background_persistence(mask:np.ndarray, bg_labels:np.ndarray,
                                    bg_df:pd.DataFrame, cfg:CorrectionConfig):
    """
    Classify each background component by survival under small morphological
    closings of the occupied-tissue mask. The biological segmentation is never
    overwritten; this produces mechanics-only metadata.
    """
    tissue=(mask>0)
    surv={}
    for r in cfg.persistence_radii:
        closed=ndimage.binary_closing(tissue,structure=disk(int(r)))
        remaining=~closed
        # overlap count of each original component with remaining background
        ids=bg_df.background_component.astype(int).to_numpy() if len(bg_df) else np.array([],int)
        if len(ids):
            rem=ndimage.sum(remaining,bg_labels,ids)
            sizes=bg_df.pixels.astype(float).to_numpy()
            surv[int(r)]=np.divide(rem,np.maximum(sizes,1.0))

    rows=[]
    for ix,r in enumerate(bg_df.itertuples()):
        bc=int(r.background_component); kind=str(r.kind); area=int(r.pixels)
        vals={f"survival_r{k}":float(surv[k][ix]) for k in surv}
        s1=vals.get("survival_r1",1.0)
        s2=vals.get("survival_r2",s1)
        s3=vals.get("survival_r3",s2)
        if kind=="exterior":
            cls="persistent_exterior"
        elif area<=cfg.unstable_area_max and min(s1,s2)<=cfg.unstable_survival_max:
            cls="unstable_micro_gap"
        elif s3>=cfg.persistent_survival_min:
            cls="persistent_internal_boundary"
        else:
            cls="ambiguous_internal_gap"
        rows.append({
            "background_component":bc,
            "kind":kind,
            "pixels":area,
            **vals,
            "mechanical_boundary_class":cls,
        })
    return pd.DataFrame(rows)


def junction_incidence_counts(J:pd.DataFrame):
    c=Counter()
    for r in J.itertuples():
        for e in r.incident_interfaces:
            c[int(e)]+=1
    return c


def corrected_objects(E:pd.DataFrame,J:pd.DataFrame,persistence:pd.DataFrame,
                      cfg:CorrectionConfig,mode:str):
    """
    Produce mechanics-only corrected interface/junction tables.

    baseline:
        no changes.
    persistent_boundaries:
        remove interfaces against unstable micro-gaps.
    supported_core:
        persistent-boundary correction plus removal of interface variables that
        have weak curvature information AND insufficient junction support.

    No cell labels or raster geometry are modified.
    """
    if mode=="baseline":
        return E.copy(),J.copy(),{"removed_microgap_interfaces":0,
                                 "removed_low_information_interfaces":0}

    pclass=dict(zip(persistence.background_component.astype(int),
                    persistence.mechanical_boundary_class.astype(str)))
    keep=np.ones(len(E),bool)
    micro=np.zeros(len(E),bool)
    for i,r in enumerate(E.itertuples()):
        if r.kind=="cell_boundary":
            bc=int(r.background_component)
            if pclass.get(bc)=="unstable_micro_gap":
                micro[i]=True
    keep &= ~micro

    low=np.zeros(len(E),bool)
    if mode=="supported_core":
        inc=junction_incidence_counts(J)
        for i,r in enumerate(E.itertuples()):
            if not keep[i]:
                continue
            conf=float(r.curvature_confidence)
            kap=abs(float(r.curvature))
            weak=(conf<cfg.curvature_confidence_min or kap<cfg.curvature_abs_min)
            if weak and inc.get(int(r.interface_id),0)<cfg.min_junction_incidence_for_low_curvature:
                low[i]=True
        keep &= ~low

    E2=E.loc[keep].copy()
    allowed=set(E2.interface_id.astype(int))
    rows=[]
    for r in J.itertuples():
        inc=[int(e) for e in r.incident_interfaces if int(e) in allowed]
        if len(inc)>=3:
            rows.append({
                "junction_id":int(r.junction_id),
                "x":float(r.x),"y":float(r.y),
                "incident_interfaces":inc,
                "n_regions":int(getattr(r,"n_regions",len(inc))),
            })
    J2=pd.DataFrame(rows)
    if len(J2)==0:
        J2=pd.DataFrame(columns=["junction_id","x","y","incident_interfaces","n_regions"])

    return E2,J2,{
        "removed_microgap_interfaces":int(micro.sum()),
        "removed_low_information_interfaces":int(low.sum()),
    }


def patch_rank(cells,E,J,C):
    A,b,meta=build_patch_system(cells,E,J,C,SolverConfig())
    q=rank_nullity(A)
    q["n_tension_variables"]=len(meta["eidx"])
    q["n_cell_pressure_variables"]=len(meta["cidx"])
    q["n_boundary_pressure_variables"]=len(meta["bidx"])
    q["n_young_laplace_rows"]=sum(1 for x in meta["rows"] if x[0]=="young_laplace")
    q["n_junction_rows"]=sum(1 for x in meta["rows"] if x[0] in ("junction_x","junction_y"))
    return q


def compare_patch_corrections(cells,E,J,C,persistence,cfg):
    out={}
    for mode in ("baseline","persistent_boundaries","supported_core"):
        E2,J2,audit=corrected_objects(E,J,persistence,cfg,mode)
        # restrict is handled inside build_patch_system
        q=patch_rank(cells,E2,J2,C)
        out[mode]={**q,**audit}
    base=out["baseline"]
    for mode in ("persistent_boundaries","supported_core"):
        out[mode]["nullity_reduction_vs_baseline"] = (
            base["structural_nullity"]-out[mode]["structural_nullity"]
        )
        out[mode]["rank_fraction_gain_vs_baseline"] = (
            out[mode]["structural_rank_fraction"]-base["structural_rank_fraction"]
        )
    return out
