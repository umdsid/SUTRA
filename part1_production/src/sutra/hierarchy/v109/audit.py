from __future__ import annotations
import ast, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def finite(x):
    a=np.asarray(x,float)
    return a[np.isfinite(a)]

def parse_members(v):
    if v is None:
        return []
    if isinstance(v,(list,tuple,set,np.ndarray,pd.Series)):
        return [str(x) for x in v]
    if isinstance(v,str):
        s=v.strip()
        if not s:
            return []
        try:
            x=ast.literal_eval(s)
            if isinstance(x,(list,tuple,set,np.ndarray)):
                return [str(z) for z in x]
        except Exception:
            pass
        if ";" in s:
            return [z for z in s.split(";") if z]
        if "," in s and "[" not in s:
            return [z.strip() for z in s.split(",") if z.strip()]
        return [s]
    return [str(v)]

def membership_sets(df):
    if df is None or len(df)==0:
        return {},None
    idc=next((c for c in ["supernode_id","node_id"] if c in df.columns),None)
    memc=next((c for c in ["level0_members","member_ids","members","cell_ids"] if c in df.columns),None)
    if idc is None or memc is None:
        return {},None
    out={}
    for _,r in df.iterrows():
        out[str(r[idc])]=set(parse_members(r[memc]))
    return out,memc

def partition_audit(member_map):
    """
    Audits one landmark as a Level-0 partition when explicit membership exists.
    """
    if not member_map:
        return {
            "membership_available":False,
            "supernodes":0,"total_member_assignments":0,
            "unique_level0_members":0,"duplicate_assignments":np.nan,
            "partition_is_disjoint":False
        }
    seen={}
    total=0
    for sid,S in member_map.items():
        total += len(S)
        for x in S:
            seen[x]=seen.get(x,0)+1
    dup=sum(v-1 for v in seen.values() if v>1)
    return {
        "membership_available":True,
        "supernodes":int(len(member_map)),
        "total_member_assignments":int(total),
        "unique_level0_members":int(len(seen)),
        "duplicate_assignments":int(dup),
        "partition_is_disjoint":bool(dup==0)
    }

def compare_partitions(child_map,parent_map):
    """
    Exact nesting audit between two successive frozen landmark states.
    Each child must be contained in exactly one parent. Parent mass is therefore
    non-decreasing along every lineage.
    """
    if not child_map or not parent_map:
        return pd.DataFrame(),{
            "pair_resolved":False,
            "children":0,
            "children_nested":0,
            "children_ambiguous":0,
            "children_orphaned":0,
            "mass_monotone_all":False
        }
    rows=[]
    parents=list(parent_map.items())
    for cid,C in child_map.items():
        containing=[]
        overlaps=[]
        for pid,P in parents:
            inter=len(C&P)
            if inter:
                overlaps.append((pid,inter,len(P)))
            if C.issubset(P):
                containing.append((pid,len(P)))
        if len(containing)==1:
            pid,pmass=containing[0]
            nested=True;amb=False;orph=False
        elif len(containing)>1:
            pid,pmass=containing[0]
            nested=False;amb=True;orph=False
        else:
            pid=None;pmass=np.nan
            nested=False;amb=False;orph=True
        rows.append({
            "child_supernode_id":cid,
            "child_mass":int(len(C)),
            "parent_supernode_id":pid,
            "parent_mass":pmass,
            "nested_in_exactly_one_parent":nested,
            "ambiguous_parent":amb,
            "orphan_child":orph,
            "mass_monotone":bool(nested and pmass>=len(C)),
            "n_parent_overlaps":int(len(overlaps))
        })
    d=pd.DataFrame(rows)
    summ={
        "pair_resolved":True,
        "children":int(len(d)),
        "children_nested":int(d.nested_in_exactly_one_parent.sum()),
        "children_ambiguous":int(d.ambiguous_parent.sum()),
        "children_orphaned":int(d.orphan_child.sum()),
        "mass_monotone_all":bool(d.mass_monotone.all()) if len(d) else False
    }
    return d,summ

def dominant_by_mass(state,mass_fraction=0.8):
    """
    Select the smallest set of largest supernodes containing the requested
    fraction of Level-0 mass.
    """
    if state is None or len(state)==0:
        return pd.DataFrame()
    idc=next((c for c in ["supernode_id","node_id"] if c in state.columns),None)
    memc=next((c for c in ["level0_members","member_ids","members","cell_ids"] if c in state.columns),None)
    massc=next((c for c in ["n_cells","cell_count","size","mass"] if c in state.columns),None)
    if idc is None:
        return pd.DataFrame()
    if massc:
        mass=pd.to_numeric(state[massc],errors="coerce").to_numpy(float)
    elif memc:
        mass=np.array([len(parse_members(x)) for x in state[memc]],float)
    else:
        return pd.DataFrame()
    d=pd.DataFrame({
        "supernode_id":state[idc].astype(str).to_numpy(),
        "mass":mass,
        "row_index":np.arange(len(state),dtype=int)
    })
    d=d[np.isfinite(d.mass)&(d.mass>0)].sort_values("mass",ascending=False).reset_index(drop=True)
    if len(d)==0:
        return d
    d["mass_fraction"]=d.mass/d.mass.sum()
    d["cumulative_mass_fraction"]=d.mass_fraction.cumsum()
    k=int(np.searchsorted(d.cumulative_mass_fraction.to_numpy(float),float(mass_fraction),side="left")+1)
    d["dominant_K"]=False
    d.loc[:k-1,"dominant_K"]=True
    d["dominant_target_fraction"]=float(mass_fraction)
    return d

def schema_names(path):
    try:
        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return []

def find_expression_table(project,sample,cfg):
    roots=[]
    for rel in cfg["expression_search_roots"]:
        p=project/rel
        if p.exists():
            roots.append(p)
    cand=[]
    for rr in roots:
        for p in rr.rglob("*.parquet"):
            cols=schema_names(p)
            if not cols:
                continue
            idc=next((c for c in ["cell_id","cell","barcode","cell_index"] if c in cols),None)
            if not idc:
                continue
            expr=[c for c in cols if c.startswith("gene_") or c.startswith("expr_") or c.startswith("expression_")]
            if len(expr)>=int(cfg["min_expression_features"]):
                cand.append((len(expr),p,idc,expr))
    if not cand:
        return None
    cand.sort(key=lambda z:z[0],reverse=True)
    return cand[0]

def find_coordinate_table(project,sample,cfg):
    roots=[]
    for rel in cfg["coordinate_search_roots"]:
        p=project/rel
        if p.exists():
            roots.append(p)
    cand=[]
    for rr in roots:
        for p in rr.rglob("*.parquet"):
            cols=schema_names(p)
            if not cols:
                continue
            idc=next((c for c in ["cell_id","cell","barcode","cell_index"] if c in cols),None)
            xc=next((c for c in ["x","x_coord","centroid_x","cell_centroid_x"] if c in cols),None)
            yc=next((c for c in ["y","y_coord","centroid_y","cell_centroid_y"] if c in cols),None)
            if idc and xc and yc:
                cand.append((p,idc,xc,yc))
    return cand[0] if cand else None

def expression_coherence(state,dominant,expr_source,cfg):
    if state is None or len(state)==0 or expr_source is None or len(dominant)==0:
        return pd.DataFrame()
    idc=next((c for c in ["supernode_id","node_id"] if c in state.columns),None)
    memc=next((c for c in ["level0_members","member_ids","members","cell_ids"] if c in state.columns),None)
    if idc is None or memc is None:
        return pd.DataFrame()
    _,path,cell_id_col,expr_cols=expr_source
    expr_cols=expr_cols[:int(cfg["max_expression_features"])]
    d=pd.read_parquet(path,columns=[cell_id_col]+expr_cols)
    d[cell_id_col]=d[cell_id_col].astype(str)
    d=d.drop_duplicates(cell_id_col).set_index(cell_id_col)
    rows=[]
    dom=set(dominant.loc[dominant.dominant_K,"supernode_id"].astype(str))
    for _,r in state.iterrows():
        sid=str(r[idc])
        if sid not in dom:
            continue
        members=parse_members(r[memc])
        present=[m for m in members if m in d.index]
        rec={
            "supernode_id":sid,
            "members_declared":int(len(members)),
            "members_with_expression":int(len(present)),
            "expression_source":str(path),
            "expression_features":int(len(expr_cols))
        }
        if len(present)>=2:
            X=d.loc[present,expr_cols].apply(pd.to_numeric,errors="coerce").to_numpy(float)
            good=np.isfinite(X).sum(axis=0)>=2
            X=X[:,good]
            if X.shape[1]:
                mu=np.nanmean(X,axis=0)
                sq=(X-mu[None,:])**2
                vals=sq[np.isfinite(sq)]
                rec["within_expression_mse"]=float(vals.mean()) if len(vals) else np.nan
                rec["within_expression_median_variance"]=float(np.nanmedian(np.nanvar(X,axis=0,ddof=0)))
            else:
                rec["within_expression_mse"]=np.nan
                rec["within_expression_median_variance"]=np.nan
        else:
            rec["within_expression_mse"]=np.nan
            rec["within_expression_median_variance"]=np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

def spatial_coherence(state,dominant,coord_source):
    if state is None or len(state)==0 or coord_source is None or len(dominant)==0:
        return pd.DataFrame()
    idc=next((c for c in ["supernode_id","node_id"] if c in state.columns),None)
    memc=next((c for c in ["level0_members","member_ids","members","cell_ids"] if c in state.columns),None)
    if idc is None or memc is None:
        return pd.DataFrame()
    path,cell_id_col,xc,yc=coord_source
    d=pd.read_parquet(path,columns=[cell_id_col,xc,yc])
    d[cell_id_col]=d[cell_id_col].astype(str)
    d=d.drop_duplicates(cell_id_col).set_index(cell_id_col)
    dom=set(dominant.loc[dominant.dominant_K,"supernode_id"].astype(str))
    rows=[]
    for _,r in state.iterrows():
        sid=str(r[idc])
        if sid not in dom:
            continue
        members=parse_members(r[memc])
        present=[m for m in members if m in d.index]
        rec={
            "supernode_id":sid,
            "members_declared":int(len(members)),
            "members_with_coordinates":int(len(present)),
            "coordinate_source":str(path)
        }
        if len(present)>=2:
            x=pd.to_numeric(d.loc[present,xc],errors="coerce").to_numpy(float)
            y=pd.to_numeric(d.loc[present,yc],errors="coerce").to_numpy(float)
            q=np.isfinite(x)&np.isfinite(y)
            if q.sum()>=2:
                xm=float(x[q].mean()); ym=float(y[q].mean())
                rr=(x[q]-xm)**2+(y[q]-ym)**2
                rec["spatial_rms_radius"]=float(np.sqrt(np.mean(rr)))
                rec["spatial_extent_x"]=float(x[q].max()-x[q].min())
                rec["spatial_extent_y"]=float(y[q].max()-y[q].min())
                cov=np.cov(np.column_stack([x[q],y[q]]),rowvar=False,bias=True)
                ev=np.linalg.eigvalsh(cov)
                rec["spatial_anisotropy_ratio"]=float(ev[-1]/max(ev[0],1e-12)) if len(ev)==2 else np.nan
            else:
                rec["spatial_rms_radius"]=np.nan
        else:
            rec["spatial_rms_radius"]=np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

def state_path_from_manifest(manifest,candidate_landmark):
    if manifest is None or len(manifest)==0 or "candidate_landmark" not in manifest.columns:
        return None
    q=manifest[pd.to_numeric(manifest.candidate_landmark,errors="coerce")==int(candidate_landmark)]
    if len(q)==0:
        return None
    r=q.iloc[0]
    p=r.get("node_state_path",r.get("node_state_source",None))
    if p is None or (isinstance(p,float) and np.isnan(p)):
        return None
    return Path(str(p))

def read_state(path):
    if path is None or not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None
