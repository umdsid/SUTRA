from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.geometry.base import BaseGeometry


def _line_parts(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, LineString):
        return [g]
    if isinstance(g, MultiLineString):
        return list(g.geoms)
    if isinstance(g, GeometryCollection):
        out=[]
        for h in g.geoms:
            out.extend(_line_parts(h))
        return out
    return []


def reconstruct_interface_arc(gi: BaseGeometry, gj: BaseGeometry, epsilon: float):
    """
    Return the longest measured boundary arc of gi that is supported by the
    epsilon-neighborhood of gj's measured boundary.
    """
    try:
        if epsilon == 0:
            g = gi.boundary.intersection(gj.boundary)
        else:
            g = gi.boundary.intersection(gj.boundary.buffer(epsilon))
    except Exception:
        return None
    parts=[x for x in _line_parts(g) if x.length > 0]
    if not parts:
        return None
    return max(parts, key=lambda x:x.length)


def materialize_interface_geometry(certified, polygons, epsilon):
    rows=[]
    for r in certified.itertuples(index=False):
        a,b=str(r.cell_i),str(r.cell_j)
        gi,gj=polygons.get(a),polygons.get(b)
        if gi is None or gj is None:
            continue
        arc=reconstruct_interface_arc(gi,gj,epsilon)
        if arc is None or arc.length <= 0:
            continue
        coords=list(arc.coords)
        if len(coords)<2:
            continue
        p0=np.asarray(coords[0],float)
        p1=np.asarray(coords[-1],float)
        chord=p1-p0
        clen=float(np.linalg.norm(chord))
        if not np.isfinite(clen) or clen<=0:
            continue
        t=chord/clen
        ci=np.array([gi.centroid.x,gi.centroid.y],float)
        cj=np.array([gj.centroid.x,gj.centroid.y],float)
        n=cj-ci
        nn=float(np.linalg.norm(n))
        if nn<=0 or not np.isfinite(nn):
            continue
        n=n/nn
        # Orient tangent deterministically; tension forces at endpoints use +/- t.
        rows.append({
            "cell_i":a,"cell_j":b,
            "confidence_class":str(r.confidence_class),
            "min_gap":float(r.min_gap),
            "global_persistence_fraction":float(r.global_persistence_fraction),
            "interface_length":float(arc.length),
            "endpoint0_x":float(p0[0]),"endpoint0_y":float(p0[1]),
            "endpoint1_x":float(p1[0]),"endpoint1_y":float(p1[1]),
            "tangent_x":float(t[0]),"tangent_y":float(t[1]),
            "normal_i_to_j_x":float(n[0]),"normal_i_to_j_y":float(n[1]),
        })
    return pd.DataFrame(rows)


class _UF:
    def __init__(self,n):
        self.p=list(range(n)); self.r=[0]*n
    def find(self,a):
        while self.p[a]!=a:
            self.p[a]=self.p[self.p[a]]
            a=self.p[a]
        return a
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.r[a]<self.r[b]: a,b=b,a
        self.p[b]=a
        if self.r[a]==self.r[b]: self.r[a]+=1


def cluster_endpoints(interface_geom: pd.DataFrame, merge_tolerance: float):
    if interface_geom.empty:
        return interface_geom.copy(), pd.DataFrame()

    pts=[]
    owners=[]
    for eidx,r in enumerate(interface_geom.itertuples(index=False)):
        pts.append((float(r.endpoint0_x),float(r.endpoint0_y))); owners.append((eidx,0))
        pts.append((float(r.endpoint1_x),float(r.endpoint1_y))); owners.append((eidx,1))
    pts=np.asarray(pts,float)

    uf=_UF(len(pts))
    tree=cKDTree(pts)
    for a,b in tree.query_pairs(r=merge_tolerance):
        uf.union(int(a),int(b))

    roots={}
    labels=np.empty(len(pts),dtype=int)
    for k in range(len(pts)):
        rt=uf.find(k)
        if rt not in roots: roots[rt]=len(roots)
        labels[k]=roots[rt]

    geom=interface_geom.copy()
    j0=np.empty(len(geom),dtype=int)
    j1=np.empty(len(geom),dtype=int)
    for point_idx,(eidx,end) in enumerate(owners):
        if end==0:j0[eidx]=labels[point_idx]
        else:j1[eidx]=labels[point_idx]
    geom["junction0"]=j0
    geom["junction1"]=j1

    jrows=[]
    for jid in sorted(set(labels.tolist())):
        idx=np.where(labels==jid)[0]
        incident=set(owners[k][0] for k in idx)
        jrows.append({
            "junction_id":int(jid),
            "x":float(np.mean(pts[idx,0])),
            "y":float(np.mean(pts[idx,1])),
            "n_endpoint_observations":int(len(idx)),
            "n_incident_interfaces":int(len(incident)),
        })
    return geom,pd.DataFrame(jrows)
