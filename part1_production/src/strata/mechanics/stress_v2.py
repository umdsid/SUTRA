from __future__ import annotations
import math
import numpy as np
import pandas as pd


def derive_stress(cells, edges, pressure):
    area={}
    for r in cells.itertuples(index=False):
        a=float(getattr(r,"cell_area",np.nan))
        area[str(r.cell_id)] = a if np.isfinite(a) and a>0 else math.nan
    pmap=dict(zip(pressure.cell_id.astype(str),pressure.pressure_like.astype(float)))
    accum={c:np.zeros((2,2),float) for c in pmap}

    for r in edges.itertuples(index=False):
        tau=float(r.tension_like); L=float(r.interface_length)
        tx,ty=float(r.tangent_x),float(r.tangent_y)
        if not np.isfinite([tau,L,tx,ty]).all(): continue
        tt=np.array([[tx*tx,tx*ty],[tx*ty,ty*ty]],float)
        for c in (str(r.cell_i),str(r.cell_j)):
            if c in accum: accum[c]+=tau*L*tt

    rows=[]
    for c,p in pmap.items():
        A=area.get(c,math.nan)
        if np.isfinite(A) and A>0:
            s=-p*np.eye(2)+accum[c]/A
            vals,vecs=np.linalg.eigh(s)
            order=np.argsort(vals)[::-1]
            vals=vals[order]; v=vecs[:,order[0]]
            s1,s2=float(vals[0]),float(vals[1])
            anis=abs(s1-s2)/(abs(s1)+abs(s2)+1e-12)
            theta=math.atan2(float(v[1]),float(v[0]))
            valid=True
        else:
            s=np.full((2,2),np.nan); s1=s2=anis=theta=np.nan; valid=False
        rows.append({
            "cell_id":c,
            "stress_xx":float(s[0,0]),"stress_xy":float(s[0,1]),"stress_yy":float(s[1,1]),
            "principal_stress_1":s1,"principal_stress_2":s2,
            "stress_anisotropy":anis,"principal_orientation_rad":theta,
            "stress_valid":valid
        })
    return pd.DataFrame(rows)
