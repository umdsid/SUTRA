from __future__ import annotations
import gzip,json
from pathlib import Path
import numpy as np
import pandas as pd
from shapely.geometry import shape


def read_geojson(path: Path):
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt") as f:
        obj=json.load(f)
    rows=[]
    for feat in obj.get("features",[]):
        props=feat.get("properties",{}) or {}
        geom=shape(feat["geometry"])
        rows.append({"properties":props,"geometry":geom})
    return rows


def proseg_polygons(path: Path):
    rows=read_geojson(path)
    out=[]
    for i,r in enumerate(rows):
        props=r["properties"]
        cid=None
        for k in ("cell","cell_id","id","label"):
            if k in props:
                cid=str(props[k]);break
        out.append({"proseg_id":cid if cid is not None else str(i+1),"geometry":r["geometry"]})
    return out


def match_proseg_to_original(proseg, cells):
    """
    Match Proseg territories to original cells by nearest centroid, then greedily
    enforce uniqueness. This avoids assuming Proseg's output IDs equal Xenium IDs.
    """
    from scipy.spatial import cKDTree
    c=cells.copy()
    c["cell_id"]=c.cell_id.astype(str)
    xy=c[["x_centroid","y_centroid"]].to_numpy(float)
    tree=cKDTree(xy)
    cand=[]
    for i,r in enumerate(proseg):
        p=r["geometry"]
        cen=np.array([p.centroid.x,p.centroid.y],float)
        d,idx=tree.query(cen,k=min(5,len(c)))
        for dd,jj in zip(np.atleast_1d(d),np.atleast_1d(idx)):
            cand.append((float(dd),i,int(jj)))
    cand.sort()
    usedp=set();usedc=set();mapping={}
    for d,i,j in cand:
        if i in usedp or j in usedc: continue
        usedp.add(i);usedc.add(j)
        mapping[i]=str(c.iloc[j].cell_id)
    return mapping
