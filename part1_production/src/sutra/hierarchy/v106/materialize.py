from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def finite(x):
    a=np.asarray(x,float)
    return a[np.isfinite(a)]

def minmax01(x):
    a=np.asarray(x,float)
    out=np.full(len(a),np.nan)
    q=np.isfinite(a)
    if not q.any(): return out
    lo=float(np.min(a[q])); hi=float(np.max(a[q]))
    out[q]=1.0 if hi-lo<=1e-12 else (a[q]-lo)/(hi-lo)
    return out

def assign_landmark_ids(front,sample):
    x=front.sort_values("minimum_removed_fraction").copy().reset_index(drop=True)
    x["landmark_order"]=np.arange(1,len(x)+1)
    x["landmark_id"]=[f"{sample}:L{int(k)}" for k in x.landmark_order]
    if "pareto_layer" not in x.columns:
        x["pareto_layer"]=1
    x["pareto_layer"]=x["pareto_layer"].fillna(1).astype(int)
    x["principal_landmark"]=x["pareto_layer"].eq(1)
    return x

def compute_lsi(x,cfg):
    y=x.copy()
    life=minmax01(y.lifetime_ell.to_numpy(float))
    depth=minmax01(y.basin_depth.to_numpy(float))
    coh=minmax01(y.cross_block_coherence.to_numpy(float))
    layer=y.pareto_layer.to_numpy(float)
    ls=np.full(len(layer),np.nan)
    q=np.isfinite(layer)
    if q.any():
        ls[q]=1.0/np.maximum(layer[q],1.0)
        z=finite(ls)
        if len(z) and np.max(z)-np.min(z)>1e-12:
            ls[q]=(ls[q]-np.min(z))/(np.max(z)-np.min(z))
        elif len(z):
            ls[q]=1.0
    M=np.vstack([life,depth,coh,ls]).T
    score=np.full(len(y),np.nan); nres=np.zeros(len(y),int)
    for i,row in enumerate(M):
        z=row[np.isfinite(row)]
        nres[i]=len(z)
        if len(z): score[i]=float(np.mean(z))
    y["lsi"]=score
    y["lsi_terms_resolved"]=nres
    y["lsi_is_gate"]=False
    return y

def validate_order(landmarks):
    if len(landmarks)<=1:return True
    r=landmarks.minimum_removed_fraction.to_numpy(float)
    return bool(np.all(np.diff(r)>0))

def build_hierarchy_tree(landmarks,sample):
    rows=[{
        "sample":sample,"node_id":f"{sample}:L0","node_type":"level0",
        "landmark_order":0,"parent_id":None,"previous_landmark_id":None,
        "next_landmark_id":landmarks.iloc[0].landmark_id if len(landmarks) else None,
        "minimum_removed_fraction":0.0,"principal_landmark":False,
    }]
    for i,r in landmarks.iterrows():
        rows.append({
            "sample":sample,"node_id":str(r.landmark_id),
            "node_type":"pareto_landmark","landmark_order":int(r.landmark_order),
            "parent_id":f"{sample}:L0" if i==0 else str(landmarks.iloc[i-1].landmark_id),
            "previous_landmark_id":None if i==0 else str(landmarks.iloc[i-1].landmark_id),
            "next_landmark_id":None if i==len(landmarks)-1 else str(landmarks.iloc[i+1].landmark_id),
            "minimum_removed_fraction":float(r.minimum_removed_fraction),
            "principal_landmark":bool(r.principal_landmark),
        })
    return pd.DataFrame(rows)

def graphml_text(tree):
    def esc(v):
        if v is None:return ""
        return str(v).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
    nodes=[]; edges=[]
    for r in tree.itertuples(index=False):
        nodes.append(f'    <node id="{esc(r.node_id)}"/>')
        if r.parent_id:
            edges.append(f'    <edge source="{esc(r.parent_id)}" target="{esc(r.node_id)}"/>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
            '  <graph id="STRATA" edgedefault="directed">\n'
            +"\n".join(nodes)+"\n"+"\n".join(edges)+
            '\n  </graph>\n</graphml>\n')

def discover_candidate_files(project,sample,landmark_index):
    roots=[
        project/"results"/"hierarchy_v093_full_completed_ultraslow_flow",
        project/"results"/"hierarchy_v100_adaptive_ness_flow",
        project/"results"/"hierarchy_v104_persistence_basin_landmarks"/sample,
    ]
    pats=[
        f"**/{sample}/**/*landmark*{int(landmark_index)}*.parquet",
        f"**/{sample}/**/*{int(landmark_index)}*node*.parquet",
        f"**/*landmark*{int(landmark_index)}*.parquet",
    ]
    out=[];seen=set()
    for rr in roots:
        if not rr.exists():continue
        for pat in pats:
            for p in rr.glob(pat):
                if p.is_file() and p not in seen:
                    seen.add(p);out.append(p)
    return out

def load_best_node_state(project,sample,landmark_index):
    ranked=[]
    for p in discover_candidate_files(project,sample,landmark_index):
        try:d=pd.read_parquet(p)
        except Exception:continue
        cols=set(d.columns)
        score=sum(c in cols for c in [
            "supernode_id","node_id","size","centroid_x","centroid_y",
            "expression_mean","parent_id","members","member_ids","level0_members"
        ])
        if score: ranked.append((score,len(d),p,d))
    if not ranked:return None,None
    ranked.sort(key=lambda z:(z[0],z[1]),reverse=True)
    _,_,p,d=ranked[0]
    return d,p

def summarize_node_state(df,landmark_id,source_path):
    rec={"landmark_id":landmark_id,"node_state_available":bool(df is not None),
         "node_state_source":str(source_path) if source_path else None}
    if df is None or len(df)==0:return rec
    idc=next((c for c in ["supernode_id","node_id"] if c in df.columns),None)
    rec["supernode_count"]=int(df[idc].nunique()) if idc else int(len(df))
    for c in ["size","mass","cell_count","n_cells"]:
        if c in df.columns:
            v=finite(df[c])
            if len(v):
                rec[f"{c}_mean"]=float(v.mean())
                rec[f"{c}_q95"]=float(np.quantile(v,.95))
                rec[f"{c}_max"]=float(v.max())
    xc=next((c for c in ["centroid_x","x","x_coord"] if c in df.columns),None)
    yc=next((c for c in ["centroid_y","y","y_coord"] if c in df.columns),None)
    if xc and yc:
        x=df[xc].to_numpy(float);y=df[yc].to_numpy(float);q=np.isfinite(x)&np.isfinite(y)
        if q.any():
            xm=float(x[q].mean());ym=float(y[q].mean())
            rec.update({
                "centroid_x_mean":xm,"centroid_y_mean":ym,
                "spatial_extent_x":float(x[q].max()-x[q].min()),
                "spatial_extent_y":float(y[q].max()-y[q].min()),
                "spatial_rms_radius":float(np.sqrt(np.mean((x[q]-xm)**2+(y[q]-ym)**2))),
            })
    return rec

def extract_lineage(df,landmark_id):
    if df is None or len(df)==0:return pd.DataFrame()
    idc=next((c for c in ["supernode_id","node_id"] if c in df.columns),None)
    mem=next((c for c in ["member_ids","members","level0_members","cell_ids"] if c in df.columns),None)
    par=next((c for c in ["parent_id","parent_supernode_id"] if c in df.columns),None)
    if idc is None:return pd.DataFrame()
    rows=[]
    for _,r in df.iterrows():
        rows.append({
            "landmark_id":landmark_id,"supernode_id":r.get(idc),
            "level0_members":r.get(mem) if mem else None,
            "parent_supernode_id":r.get(par) if par else None,
        })
    return pd.DataFrame(rows)

def materialize_observable_row(row,landmark_id):
    out={"landmark_id":landmark_id}
    for c,v in row.items():
        if np.isscalar(v) or v is None:
            out[str(c)]=v
    return out

def observable_groups(columns):
    groups={
        "expression":("expression_",),
        "functional":("functional_","go_","msigdb_"),
        "cellchat":("cellchat_",),
        "mechanics":("tension_","mechanics_","stress_"),
        "pressure":("pressure_",),
        "topology":("component","cycle_rank","mean_degree","max_degree","supernode_"),
        "directional_geometry":("directional_","local_geometry_"),
        "transport":("transport_",),
        "geodesics":("geodesic_",),
        "holonomy":("holonomy_",),
        "derivatives":("d1_","d2_"),
    }
    out={k:[] for k in groups}
    for c in columns:
        for k,p in groups.items():
            if any(str(c).startswith(x) for x in p):out[k].append(str(c))
    return out

def build_transition_table(x):
    rows=[]
    for i in range(len(x)-1):
        a=x.iloc[i];b=x.iloc[i+1]
        rows.append({
            "from_landmark_id":a.landmark_id,"to_landmark_id":b.landmark_id,
            "delta_removed_fraction":float(b.minimum_removed_fraction-a.minimum_removed_fraction),
            "delta_nodes":int(b.minimum_nodes-a.minimum_nodes),
            "delta_lifetime_ell":float(b.lifetime_ell-a.lifetime_ell)
                if np.isfinite(a.lifetime_ell) and np.isfinite(b.lifetime_ell) else np.nan,
            "delta_basin_depth":float(b.basin_depth-a.basin_depth)
                if np.isfinite(a.basin_depth) and np.isfinite(b.basin_depth) else np.nan,
            "delta_cross_block_coherence":float(b.cross_block_coherence-a.cross_block_coherence)
                if np.isfinite(a.cross_block_coherence) and np.isfinite(b.cross_block_coherence) else np.nan,
        })
    return pd.DataFrame(rows)
