from __future__ import annotations
import math
import numpy as np
import pandas as pd

def derive_stress(cells: pd.DataFrame, edges: pd.DataFrame, pressure: pd.DataFrame):
    area={}
    for r in cells.itertuples(index=False):
        cid=str(r.cell_id)
        a=float(getattr(r,"cell_area",np.nan))
        area[cid]=a if np.isfinite(a) and a>0 else math.nan
    pmap=dict(zip(pressure.cell_id.astype(str),pressure.pressure_like.astype(float)))
    accum={cid:np.zeros((2,2),float) for cid in pmap}
    for r in edges.itertuples(index=False):
        tau=float(r.tension_like); L=float(r.shared_support_at_chosen)
        nx=float(r.normal_i_to_j_x); ny=float(r.normal_i_to_j_y)
        if not np.isfinite([tau,L,nx,ny]).all(): continue
        nn=np.array([[nx*nx,nx*ny],[nx*ny,ny*ny]])
        for cid in (str(r.cell_i),str(r.cell_j)):
            if cid in accum: accum[cid]+=tau*L*nn
    rows=[]
    for cid,p in pmap.items():
        a=area.get(cid,math.nan)
        valid=np.isfinite(a) and a>0
        if valid:
            s=-p*np.eye(2)+accum[cid]/a
            vals,vecs=np.linalg.eigh(s)
            order=np.argsort(vals)[::-1]
            vals=vals[order]; v=vecs[:,order[0]]
            s1,s2=float(vals[0]),float(vals[1])
            anis=abs(s1-s2)/(abs(s1)+abs(s2)+1e-12)
            theta=math.atan2(float(v[1]),float(v[0]))
            sxx,sxy,syy=float(s[0,0]),float(s[0,1]),float(s[1,1])
        else:
            sxx=sxy=syy=s1=s2=anis=theta=math.nan
        rows.append(dict(
            cell_id=cid,stress_xx=sxx,stress_xy=sxy,stress_yy=syy,
            principal_stress_1=s1,principal_stress_2=s2,
            stress_anisotropy=anis,principal_orientation_rad=theta,
            stress_valid=bool(valid)
        ))
    return pd.DataFrame(rows)
