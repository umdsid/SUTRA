from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

GROUPS={
    "expression":("expression_",),
    "functional":("functional_","go_","msigdb_"),
    "cellchat":("cellchat_",),
    "mechanics":("tension_","mechanics_","stress_"),
    "pressure":("pressure_",),
    "topology":("component","cycle_rank","mean_degree","max_degree","degree_","supernode_"),
    "directional_geometry":("directional_","local_geometry_"),
    "transport":("transport_",),
    "geodesics":("geodesic_",),
    "holonomy":("holonomy_",),
    "derivatives":("d1_","d2_"),
}
KEYS=("landmark_index","candidate_landmark","nodes","removed_fraction","ell","microstep","step")

def finite(x):
    a=np.asarray(x,float)
    return a[np.isfinite(a)]

def columns_for_group(cols,group):
    pref=GROUPS[group]
    return [c for c in cols if any(str(c).startswith(p) for p in pref)]

def schema_names(path):
    try:
        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return []

def scan_parquets(project,sample,cfg):
    roots=[]
    for rel in cfg["scan_result_roots"]:
        p=project/"results"/rel
        if p.exists(): roots.append(p)
    rows=[]
    seen=set()
    for rr in roots:
        for p in rr.rglob("*.parquet"):
            if p in seen: continue
            seen.add(p)
            cols=schema_names(p)
            if not cols: continue
            group_counts={g:len(columns_for_group(cols,g)) for g in GROUPS}
            keycols=[k for k in KEYS if k in cols]
            # Do not scan huge raw cell tables that have no scale key.
            if not keycols and not any(group_counts.values()):
                continue
            rows.append({
                "path":str(p),
                "relative_path":str(p.relative_to(project)),
                "n_columns":len(cols),
                "key_columns":";".join(keycols),
                **{f"{g}_columns":int(n) for g,n in group_counts.items()},
            })
    return pd.DataFrame(rows)

def read_columns(path,columns):
    cols=schema_names(path)
    want=[c for c in columns if c in cols]
    if not want:return pd.DataFrame()
    try:return pd.read_parquet(path,columns=want)
    except Exception:return pd.DataFrame()

def exact_match_row(df,landmark):
    """
    Resolve only through explicit scale keys. No biological/state interpolation.
    """
    if df is None or len(df)==0:return None,None
    lm=int(landmark["candidate_landmark"])
    nodes=int(landmark["minimum_nodes"])
    rem=float(landmark["minimum_removed_fraction"])
    ell=float(landmark["minimum_ell"]) if np.isfinite(float(landmark.get("minimum_ell",np.nan))) else np.nan

    for c in ["landmark_index","candidate_landmark"]:
        if c in df.columns:
            q=df[pd.to_numeric(df[c],errors="coerce")==lm]
            if len(q):return q.iloc[0],f"exact_{c}"

    if "nodes" in df.columns:
        q=df[pd.to_numeric(df["nodes"],errors="coerce")==nodes]
        if len(q)==1:return q.iloc[0],"exact_nodes"
        if len(q)>1 and "removed_fraction" in q.columns:
            d=np.abs(pd.to_numeric(q["removed_fraction"],errors="coerce")-rem)
            if np.isfinite(d).any():
                j=int(np.nanargmin(d.to_numpy(float)))
                if float(d.iloc[j])<=1e-10:return q.iloc[j],"exact_nodes_removed"

    if "removed_fraction" in df.columns:
        d=np.abs(pd.to_numeric(df["removed_fraction"],errors="coerce")-rem)
        if np.isfinite(d).any():
            j=int(np.nanargmin(d.to_numpy(float)))
            if float(d.iloc[j])<=1e-10:return df.iloc[j],"exact_removed_fraction"

    if "ell" in df.columns and np.isfinite(ell):
        d=np.abs(pd.to_numeric(df["ell"],errors="coerce")-ell)
        if np.isfinite(d).any():
            j=int(np.nanargmin(d.to_numpy(float)))
            if float(d.iloc[j])<=1e-10:return df.iloc[j],"exact_ell"
    return None,None

def resolve_group_source(inventory,group,landmark):
    count_col=f"{group}_columns"
    if count_col not in inventory.columns:return None
    cand=inventory[inventory[count_col]>0].copy()
    if len(cand)==0:return None

    # Prefer richer block coverage, then tables containing explicit landmark index.
    cand["has_landmark_key"]=cand.key_columns.astype(str).str.contains("landmark_index|candidate_landmark")
    cand=cand.sort_values([count_col,"has_landmark_key","n_columns"],ascending=[False,False,False])

    for r in cand.itertuples(index=False):
        path=Path(r.path)
        cols=schema_names(path)
        gcols=columns_for_group(cols,group)
        keycols=[k for k in KEYS if k in cols]
        df=read_columns(path,keycols+gcols)
        row,method=exact_match_row(df,landmark)
        if row is None:continue
        vals={}
        nfinite=0
        for c in gcols:
            v=row.get(c,np.nan)
            vals[c]=v
            try:
                if np.isscalar(v) and pd.notna(v): nfinite+=1
            except Exception:
                pass
        if nfinite==0 and gcols:
            # Preserve the row, but do not claim scientific resolution.
            continue
        return {
            "group":group,"path":str(path),"method":method,
            "columns":gcols,"values":vals,"resolved_columns":int(nfinite)
        }
    return None

def v104_state_manifest(project,sample):
    """
    v1.0.4 was the stage that successfully located all basin node states.
    Reuse those exact paths rather than re-guessing.
    """
    p=project/"results"/"hierarchy_v104_persistence_basin_landmarks"/sample/"landmark_spatial_gene_state.parquet"
    if not p.exists():return pd.DataFrame()
    try:return pd.read_parquet(p)
    except Exception:return pd.DataFrame()

def state_path_for_candidate(manifest,candidate_landmark):
    if manifest is None or len(manifest)==0:return None
    if "candidate_landmark" not in manifest.columns:return None
    q=manifest[pd.to_numeric(manifest["candidate_landmark"],errors="coerce")==int(candidate_landmark)]
    if len(q)==0:return None
    r=q.iloc[0]
    avail=bool(r.get("node_state_available",False))
    path=r.get("node_state_path",r.get("node_state_source",None))
    if not avail or path is None or (isinstance(path,float) and np.isnan(path)):return None
    return Path(str(path))

def read_state(path):
    if path is None or not path.exists():return None
    try:return pd.read_parquet(path)
    except Exception:return None

def state_summary(df,landmark_id,path):
    rec={"landmark_id":landmark_id,"node_state_available":bool(df is not None),
         "node_state_source":str(path) if path else None}
    if df is None or len(df)==0:return rec
    idc=next((c for c in ["supernode_id","node_id"] if c in df.columns),None)
    rec["supernode_count"]=int(df[idc].nunique()) if idc else int(len(df))
    for c in ["size","mass","cell_count","n_cells"]:
        if c in df.columns:
            v=finite(df[c])
            if len(v):
                rec[f"{c}_mean"]=float(np.mean(v));rec[f"{c}_q95"]=float(np.quantile(v,.95))
                rec[f"{c}_max"]=float(np.max(v))
    xc=next((c for c in ["centroid_x","x","x_coord"] if c in df.columns),None)
    yc=next((c for c in ["centroid_y","y","y_coord"] if c in df.columns),None)
    if xc and yc:
        x=pd.to_numeric(df[xc],errors="coerce").to_numpy(float)
        y=pd.to_numeric(df[yc],errors="coerce").to_numpy(float)
        q=np.isfinite(x)&np.isfinite(y)
        if q.any():
            xm=float(x[q].mean());ym=float(y[q].mean())
            rec.update({
                "centroid_x_mean":xm,"centroid_y_mean":ym,
                "spatial_extent_x":float(x[q].max()-x[q].min()),
                "spatial_extent_y":float(y[q].max()-y[q].min()),
                "spatial_rms_radius":float(np.sqrt(np.mean((x[q]-xm)**2+(y[q]-ym)**2))),
            })
    return rec

def lineage_from_state(df,landmark_id):
    if df is None or len(df)==0:return pd.DataFrame()
    idc=next((c for c in ["supernode_id","node_id"] if c in df.columns),None)
    mem=next((c for c in ["member_ids","members","level0_members","cell_ids"] if c in df.columns),None)
    par=next((c for c in ["parent_id","parent_supernode_id"] if c in df.columns),None)
    if idc is None:return pd.DataFrame()
    rows=[]
    for _,r in df.iterrows():
        rows.append({
            "landmark_id":landmark_id,
            "supernode_id":r.get(idc),
            "level0_members":r.get(mem) if mem else None,
            "parent_supernode_id":r.get(par) if par else None,
        })
    return pd.DataFrame(rows)

def materialize_landmark_groups(inventory,landmark):
    records=[];wide={"landmark_id":landmark["landmark_id"]}
    for group in GROUPS:
        res=resolve_group_source(inventory,group,landmark)
        if res is None:
            records.append({
                "landmark_id":landmark["landmark_id"],"group":group,
                "resolved":False,"source_path":None,"resolution_method":None,
                "resolved_columns":0
            })
            continue
        records.append({
            "landmark_id":landmark["landmark_id"],"group":group,
            "resolved":True,"source_path":res["path"],"resolution_method":res["method"],
            "resolved_columns":res["resolved_columns"]
        })
        for c,v in res["values"].items():
            # prevent same-name collision between multiple groups
            wide[c]=v
    return pd.DataFrame(records),wide

def completeness_matrix(provenance):
    if len(provenance)==0:return pd.DataFrame()
    p=provenance.copy()
    return p.pivot(index="landmark_id",columns="group",values="resolved").reset_index()

def sample_complete(provenance,state_summary_df,cfg):
    required=list(cfg["required_observable_groups"])
    if len(provenance)==0:return False
    for lid,g in provenance.groupby("landmark_id"):
        m={str(r.group):bool(r.resolved) for r in g.itertuples(index=False)}
        if any(not m.get(x,False) for x in required):
            return False
    if bool(cfg["require_node_state"]):
        if len(state_summary_df)==0 or not bool(state_summary_df.node_state_available.all()):
            return False
    return True
