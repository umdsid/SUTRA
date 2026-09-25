from __future__ import annotations
import ast, json, math
from pathlib import Path
import numpy as np
import pandas as pd

def parse_members(v):
    if v is None:
        return []
    if isinstance(v,(list,tuple,set,np.ndarray,pd.Series)):
        return [str(x) for x in v]
    if isinstance(v,str):
        s=v.strip()
        if not s:return []
        try:
            x=ast.literal_eval(s)
            if isinstance(x,(list,tuple,set,np.ndarray)):
                return [str(z) for z in x]
        except Exception:
            pass
        if ";" in s:return [z for z in s.split(";") if z]
        if "," in s and "[" not in s:return [z.strip() for z in s.split(",") if z.strip()]
        return [s]
    return [str(v)]

def id_member_columns(df):
    idc=next((c for c in ["supernode_id","node_id","cluster_id","state_id"] if c in df.columns),None)
    memc=next((c for c in ["level0_members","member_ids","members","cell_ids"] if c in df.columns),None)
    return idc,memc

def partition_metrics(df):
    idc,memc=id_member_columns(df)
    if idc is None or memc is None or len(df)==0:
        return {
            "rows":int(len(df)),"membership_available":False,
            "unique_members":0,"total_assignments":0,
            "duplicates":np.nan,"disjoint":False
        }
    counts={}
    total=0
    for v in df[memc]:
        S=set(parse_members(v))
        total += len(S)
        for x in S:counts[x]=counts.get(x,0)+1
    dup=sum(n-1 for n in counts.values() if n>1)
    return {
        "rows":int(len(df)),"membership_available":True,
        "unique_members":int(len(counts)),"total_assignments":int(total),
        "duplicates":int(dup),"disjoint":bool(dup==0)
    }

def _bool_series(s):
    if pd.api.types.is_bool_dtype(s):return s.fillna(False).astype(bool)
    z=s.astype(str).str.strip().str.lower()
    true=z.isin(["1","true","yes","active","current","alive","kept"])
    false=z.isin(["0","false","no","inactive","retired","dead","merged","historical"])
    if (true|false).mean()>=0.8:return true
    return None

def candidate_active_masks(df,target_step=None,target_level=None):
    """
    Generate only masks justified by explicit schema semantics.
    No score-based arbitrary row dropping.
    """
    out=[("all_rows",np.ones(len(df),dtype=bool),"baseline")]
    n=len(df)

    for c in df.columns:
        lc=c.lower()
        if lc in {"active","is_active","current","is_current","alive","is_alive","kept","is_kept"}:
            b=_bool_series(df[c])
            if b is not None:out.append((f"{c}=true",b.to_numpy(bool),"explicit_boolean"))

        if lc in {"status","state_status","node_status"}:
            z=df[c].astype(str).str.strip().str.lower()
            for val in ["active","current","alive","kept"]:
                if (z==val).any():
                    out.append((f"{c}={val}",(z==val).to_numpy(bool),"explicit_status"))

    # Explicit birth/death interval semantics.
    births=[c for c in df.columns if c.lower() in {"birth_step","created_step","start_step","born_at","birth_level","created_level"}]
    deaths=[c for c in df.columns if c.lower() in {"death_step","retired_step","end_step","merged_step","death_level","retired_level"}]
    if births and deaths and target_step is not None:
        for bc in births:
            for dc in deaths:
                b=pd.to_numeric(df[bc],errors="coerce")
                d=pd.to_numeric(df[dc],errors="coerce")
                mask=(b<=target_step)&(d.isna()|(d>target_step))
                out.append((f"{bc}<={target_step}<{dc}",mask.to_numpy(bool),"birth_death_interval"))

    # Exact snapshot selectors.
    for c in df.columns:
        lc=c.lower()
        if target_step is not None and lc in {"step","microstep","snapshot_step","state_step"}:
            v=pd.to_numeric(df[c],errors="coerce")
            if np.isfinite(v).any():
                mask=(v==target_step)
                if mask.any():out.append((f"{c}={target_step}",mask.to_numpy(bool),"exact_step"))
        if target_level is not None and lc in {"level","hierarchy_level","snapshot_level"}:
            v=pd.to_numeric(df[c],errors="coerce")
            if np.isfinite(v).any():
                mask=(v==target_level)
                if mask.any():out.append((f"{c}={target_level}",mask.to_numpy(bool),"exact_level"))

    # Deduplicate identical masks.
    dedup=[]
    seen=set()
    for name,mask,kind in out:
        key=np.packbits(np.asarray(mask,dtype=np.uint8)).tobytes()
        if key in seen:continue
        seen.add(key);dedup.append((name,np.asarray(mask,dtype=bool),kind))
    return dedup

def evaluate_masks(df,expected_nodes,expected_level0_mass,target_step=None,target_level=None):
    rows=[]
    for name,mask,kind in candidate_active_masks(df,target_step,target_level):
        sub=df.loc[mask].copy()
        m=partition_metrics(sub)
        node_match=(m["rows"]==int(expected_nodes))
        mass_match=(m["unique_members"]==int(expected_level0_mass)) if expected_level0_mass is not None else False
        exact_partition=bool(m["membership_available"] and m["disjoint"] and node_match and mass_match and m["total_assignments"]==m["unique_members"])
        rows.append({
            "candidate":name,"candidate_kind":kind,
            "selected_rows":m["rows"],"expected_nodes":int(expected_nodes),
            "unique_members":m["unique_members"],"expected_level0_mass":expected_level0_mass,
            "total_assignments":m["total_assignments"],"duplicates":m["duplicates"],
            "disjoint":m["disjoint"],"node_match":node_match,"mass_match":mass_match,
            "exact_partition":exact_partition
        })
    return pd.DataFrame(rows)

def choose_unique_partition(eval_df):
    if eval_df is None or len(eval_df)==0:return None
    q=eval_df[eval_df.exact_partition.astype(bool)]
    if len(q)!=1:return None
    return q.iloc[0].to_dict()

def apply_named_candidate(df,name,target_step=None,target_level=None):
    for cname,mask,kind in candidate_active_masks(df,target_step,target_level):
        if cname==name:return df.loc[mask].copy()
    return None

def map_partition(df):
    idc,memc=id_member_columns(df)
    if idc is None or memc is None:return {}
    return {str(r[idc]):set(parse_members(r[memc])) for _,r in df.iterrows()}

def compare_nested(child,parent):
    if not child or not parent:return pd.DataFrame(),False
    rows=[]
    for cid,C in child.items():
        containing=[(pid,P) for pid,P in parent.items() if C.issubset(P)]
        rows.append({
            "child_supernode_id":cid,"child_mass":len(C),
            "n_exact_parents":len(containing),
            "parent_supernode_id":containing[0][0] if len(containing)==1 else None,
            "parent_mass":len(containing[0][1]) if len(containing)==1 else np.nan,
            "mass_monotone":bool(len(containing)==1 and len(containing[0][1])>=len(C))
        })
    d=pd.DataFrame(rows)
    ok=bool(len(d) and (d.n_exact_parents==1).all() and d.mass_monotone.all())
    return d,ok

def state_schema_record(path,df):
    return {
        "path":str(path),
        "rows":int(len(df)),
        "columns":list(map(str,df.columns)),
        "dtypes":{str(c):str(df[c].dtype) for c in df.columns},
    }

def inventory_level0_sources(project,cfg):
    """
    Inventory likely Level-0 expression/coordinate stores without requiring a
    particular storage format.
    """
    exts=set(cfg["source_extensions"])
    rows=[]
    roots=[]
    for rel in cfg["source_search_roots"]:
        p=project/rel
        if p.exists():roots.append(p)
    seen=set()
    for rr in roots:
        for p in rr.rglob("*"):
            if not p.is_file() or p in seen:continue
            seen.add(p)
            if p.suffix.lower() not in exts and not any(str(p).lower().endswith(x) for x in exts):
                continue
            rec={"path":str(p),"relative_path":str(p.relative_to(project)),
                 "suffix":p.suffix.lower(),"size_bytes":int(p.stat().st_size)}
            low=p.name.lower()
            rec["name_expression_hint"]=any(x in low for x in ["matrix","expression","feature","transcript","count"])
            rec["name_coordinate_hint"]=any(x in low for x in ["cell","coordinate","centroid","boundary","spatial"])
            # Lightweight schema inspection only.
            if p.suffix.lower()==".parquet":
                try:
                    import pyarrow.parquet as pq
                    rec["schema"]=list(pq.ParquetFile(p).schema_arrow.names)
                except Exception:
                    rec["schema"]=[]
            elif p.suffix.lower() in {".h5",".hdf5",".h5ad"}:
                try:
                    import h5py
                    with h5py.File(p,"r") as h:
                        names=[]
                        h.visit(lambda x:names.append(x))
                        rec["schema"]=names[:500]
                except Exception:
                    rec["schema"]=[]
            elif p.suffix.lower()==".npz":
                try:
                    z=np.load(p,allow_pickle=False)
                    rec["schema"]=list(z.files)
                except Exception:
                    rec["schema"]=[]
            else:
                rec["schema"]=[]
            rows.append(rec)
    return pd.DataFrame(rows)

def classify_source_inventory(inv):
    if inv is None or len(inv)==0:return inv
    out=inv.copy()
    def _schema_text(r):
        schema=getattr(r,"schema",[])
        if schema is None:
            schema=[]
        # Parquet round-trips can return list-like, ndarray-like, or string
        # schema values. Normalize without assuming dict semantics.
        if isinstance(schema,str):
            parts=[schema]
        else:
            try:
                parts=list(schema)
            except TypeError:
                parts=[schema]
        return " ".join(map(str,parts)).lower()+" "+str(getattr(r,"relative_path","")).lower()
    def score_expr(r):
        names=_schema_text(r)
        return sum(x in names for x in ["matrix","data","indices","indptr","features","genes","expression","count"])
    def score_coord(r):
        names=_schema_text(r)
        return sum(x in names for x in ["x","y","centroid","coordinate","cells","cell_id","barcode"])
    rows=list(out.itertuples(index=False))
    out["expression_score"]=[score_expr(r) for r in rows]
    out["coordinate_score"]=[score_coord(r) for r in rows]
    return out
