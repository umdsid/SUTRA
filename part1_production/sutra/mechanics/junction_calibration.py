from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class _UF:
    def __init__(self,n):
        self.p=np.arange(n,dtype=int)
        self.r=np.zeros(n,dtype=np.int8)
    def find(self,a):
        while self.p[a]!=a:
            self.p[a]=self.p[self.p[a]]
            a=int(self.p[a])
        return a
    def union(self,a,b):
        a=self.find(int(a)); b=self.find(int(b))
        if a==b:return
        if self.r[a]<self.r[b]: a,b=b,a
        self.p[b]=a
        if self.r[a]==self.r[b]: self.r[a]+=1


def cluster_for_tolerance(interface_geom: pd.DataFrame, tol: float):
    m=len(interface_geom)
    p0=interface_geom[["endpoint0_x","endpoint0_y"]].to_numpy(float)
    p1=interface_geom[["endpoint1_x","endpoint1_y"]].to_numpy(float)
    pts=np.vstack([p0,p1])
    owners=np.concatenate([np.arange(m),np.arange(m)])

    uf=_UF(len(pts))
    tree=cKDTree(pts)
    for a,b in tree.query_pairs(r=float(tol)):
        uf.union(a,b)

    roots=np.array([uf.find(k) for k in range(len(pts))],dtype=int)
    uniq,labels=np.unique(roots,return_inverse=True)

    j0=labels[:m]
    j1=labels[m:]
    geom=interface_geom.copy()
    geom["junction0"]=j0
    geom["junction1"]=j1

    rows=[]
    degrees=[]
    for jid in range(len(uniq)):
        idx=np.flatnonzero(labels==jid)
        incident=np.unique(owners[idx])
        deg=int(len(incident))
        degrees.append(deg)
        rows.append({
            "junction_id":jid,
            "x":float(np.mean(pts[idx,0])),
            "y":float(np.mean(pts[idx,1])),
            "n_endpoint_observations":int(len(idx)),
            "n_incident_interfaces":deg,
        })
    junctions=pd.DataFrame(rows)
    d=np.asarray(degrees,dtype=int)
    stats={
        "tolerance":float(tol),
        "n_junctions":int(len(junctions)),
        "fraction_degree1":float(np.mean(d==1)) if len(d) else 1.0,
        "fraction_degree2":float(np.mean(d==2)) if len(d) else 0.0,
        "fraction_degree_ge3":float(np.mean(d>=3)) if len(d) else 0.0,
        "median_degree":float(np.median(d)) if len(d) else 0.0,
        "q95_degree":float(np.quantile(d,0.95)) if len(d) else 0.0,
        "max_degree":int(np.max(d)) if len(d) else 0,
        "endpoint_compression":float(1.0-len(junctions)/(2*max(m,1))),
    }
    return geom,junctions,stats


def calibrate_junction_tolerance(interface_geom: pd.DataFrame, gateB_epsilon: float):
    """
    Endpoint uncertainty can accumulate from both sides of a Gate-B contact.
    Therefore the physically admissible merge range is capped at 2*epsilon_B.

    We sweep from epsilon_B/2 to 2*epsilon_B and choose the earliest stable
    plateau in junction structure. If no plateau is found, choose the tolerance
    with the best degree>=3 gain before high-degree over-merging.
    """
    eps=float(gateB_epsilon)
    tolerances=np.array([0.5,0.75,1.0,1.5,2.0])*eps
    records=[]
    cache={}
    prev=None
    for t in tolerances:
        g,j,s=cluster_for_tolerance(interface_geom,float(t))
        cache[float(t)]=(g,j)
        if prev is None:
            s["rel_tri_change"]=np.inf
            s["abs_degree1_change"]=np.inf
        else:
            den=max(prev["fraction_degree_ge3"],1e-6)
            s["rel_tri_change"]=abs(s["fraction_degree_ge3"]-prev["fraction_degree_ge3"])/den
            s["abs_degree1_change"]=abs(s["fraction_degree1"]-prev["fraction_degree1"])
        # overmerge warning: highly implausible very-high-degree junction clusters
        s["overmerge_warning"]=bool(s["max_degree"]>12)
        records.append(s)
        prev=s

    sweep=pd.DataFrame(records)

    chosen=None
    # Earliest structural plateau after at least epsilon_B.
    for k in range(2,len(sweep)):
        r=sweep.iloc[k]
        if (
            r["tolerance"] >= eps
            and r["rel_tri_change"] <= 0.15
            and r["abs_degree1_change"] <= 0.03
            and not bool(r["overmerge_warning"])
        ):
            chosen=float(r["tolerance"])
            break

    if chosen is None:
        admiss=sweep[~sweep["overmerge_warning"]].copy()
        if admiss.empty:
            admiss=sweep.copy()
        # Reward true multi-interface junction recovery while penalizing degree-1
        # leftovers and excessive endpoint compression.
        score=(
            admiss["fraction_degree_ge3"]
            - 0.5*admiss["fraction_degree1"]
            - 0.10*np.maximum(admiss["endpoint_compression"]-0.75,0.0)
        )
        chosen=float(admiss.loc[score.idxmax(),"tolerance"])

    geom,junctions=cache[chosen]
    return geom,junctions,sweep,chosen
