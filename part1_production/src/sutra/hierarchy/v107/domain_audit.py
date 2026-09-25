from __future__ import annotations
import ast, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def finite(x):
    a=np.asarray(x,float)
    return a[np.isfinite(a)]

def safe_mean(x):
    a=finite(x); return float(a.mean()) if len(a) else np.nan
def safe_median(x):
    a=finite(x); return float(np.median(a)) if len(a) else np.nan
def safe_quantile(x,q):
    a=finite(x); return float(np.quantile(a,float(q))) if len(a) else np.nan

def gini(x):
    a=finite(x)
    a=a[a>=0]
    if len(a)==0 or a.sum()<=0:return np.nan
    a=np.sort(a)
    n=len(a)
    return float((2*np.sum((np.arange(1,n+1))*a)/(n*a.sum()))-(n+1)/n)

def effective_number_from_mass(mass):
    a=finite(mass)
    a=a[a>0]
    if len(a)==0:return np.nan
    p=a/a.sum()
    return float(1.0/np.sum(p*p))

def cumulative_k(mass,frac):
    a=finite(mass)
    a=a[a>0]
    if len(a)==0:return np.nan
    a=np.sort(a)[::-1]
    cs=np.cumsum(a)/a.sum()
    return int(np.searchsorted(cs,float(frac),side="left")+1)

def concentration_summary(mass):
    a=finite(mass)
    a=a[a>0]
    if len(a)==0:
        return {}
    return {
        "formal_supernodes":int(len(a)),
        "total_mass":float(a.sum()),
        "mass_median":float(np.median(a)),
        "mass_q75":float(np.quantile(a,.75)),
        "mass_q90":float(np.quantile(a,.90)),
        "mass_q95":float(np.quantile(a,.95)),
        "mass_q99":float(np.quantile(a,.99)),
        "mass_max":float(np.max(a)),
        "mass_gini":gini(a),
        "effective_domain_number":effective_number_from_mass(a),
        "K50":cumulative_k(a,.50),
        "K80":cumulative_k(a,.80),
        "K90":cumulative_k(a,.90),
    }

def parse_members(v):
    if v is None:return []
    if isinstance(v,(list,tuple,np.ndarray,pd.Series)):
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
        # conservative fallback for delimiter-separated IDs
        if ";" in s:return [z for z in s.split(";") if z]
        if "," in s and "[" not in s:return [z.strip() for z in s.split(",") if z.strip()]
        return [s]
    return [str(v)]

def infer_mass_column(df):
    for c in ["n_cells","cell_count","size","mass"]:
        if c in df.columns:
            v=pd.to_numeric(df[c],errors="coerce").to_numpy(float)
            if np.isfinite(v).any() and np.nanmax(v)>0:
                return c,v
    for c in ["level0_members","member_ids","members","cell_ids"]:
        if c in df.columns:
            v=np.array([len(parse_members(x)) for x in df[c]],float)
            if np.max(v)>0:return c,v
    return None,None

def spatial_compactness(df):
    """
    Uses only existing geometry columns. No polygons are inferred if absent.
    """
    rows=[]
    area_col=next((c for c in ["area","spatial_area","region_area"] if c in df.columns),None)
    perim_col=next((c for c in ["perimeter","boundary_length","perimeter_length"] if c in df.columns),None)
    radius_col=next((c for c in ["spatial_rms_radius","radius","rms_radius"] if c in df.columns),None)
    for i,r in df.iterrows():
        rec={"row_index":int(i)}
        if area_col and perim_col:
            A=float(r.get(area_col,np.nan)); P=float(r.get(perim_col,np.nan))
            rec["compactness_4piA_over_P2"]=float(4*np.pi*A/(P*P)) if np.isfinite(A) and np.isfinite(P) and P>0 else np.nan
        elif area_col and radius_col:
            A=float(r.get(area_col,np.nan)); R=float(r.get(radius_col,np.nan))
            rec["compactness_area_over_piR2"]=float(A/(np.pi*R*R)) if np.isfinite(A) and np.isfinite(R) and R>0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

def boundary_proxy(df):
    """
    Explicit boundary/interface evidence only. No geometry is invented.
    """
    cols=[c for c in [
        "boundary_fraction","interface_fraction","boundary_length","perimeter",
        "external_degree","interface_degree","boundary_edges"
    ] if c in df.columns]
    if not cols:return pd.DataFrame()
    out=pd.DataFrame({"row_index":np.arange(len(df),dtype=int)})
    for c in cols:
        out[c]=pd.to_numeric(df[c],errors="coerce")
    return out

def schema_names(path):
    try:return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:return []

def find_level0_expression(project,sample,cfg):
    """
    Search only configured result/data roots for an explicit cell-by-feature table
    with a cell ID and >= min_expression_features numeric/expression columns.
    """
    roots=[]
    for rel in cfg["expression_search_roots"]:
        p=project/rel
        if p.exists(): roots.append(p)
    candidates=[]
    for rr in roots:
        for p in rr.rglob("*.parquet"):
            cols=schema_names(p)
            if not cols:continue
            idc=next((c for c in ["cell_id","cell","barcode","cell_index"] if c in cols),None)
            if not idc:continue
            expr=[c for c in cols if c.startswith("gene_") or c.startswith("expr_") or c.startswith("expression_")]
            if len(expr)>=int(cfg["min_expression_features"]):
                candidates.append((len(expr),p,idc,expr))
    if not candidates:return None
    candidates.sort(key=lambda z:z[0],reverse=True)
    return candidates[0]

def expression_heterogeneity_from_members(state,expr_source,cfg):
    """
    Exact Level-0 membership only. Computes mean within-supernode squared
    dispersion across available expression features.
    """
    if state is None or len(state)==0 or expr_source is None:
        return pd.DataFrame()
    memc=next((c for c in ["level0_members","member_ids","members","cell_ids"] if c in state.columns),None)
    idc=next((c for c in ["supernode_id","node_id"] if c in state.columns),None)
    if not memc or not idc:return pd.DataFrame()

    _,path,cell_id_col,expr_cols=expr_source
    # cap only feature count, never cells/members
    expr_cols=expr_cols[:int(cfg["max_expression_features"])]
    d=pd.read_parquet(path,columns=[cell_id_col]+expr_cols)
    d[cell_id_col]=d[cell_id_col].astype(str)
    d=d.drop_duplicates(cell_id_col).set_index(cell_id_col)

    rows=[]
    for _,r in state.iterrows():
        members=parse_members(r.get(memc))
        present=[m for m in members if m in d.index]
        rec={
            "supernode_id":r.get(idc),
            "members_declared":int(len(members)),
            "members_with_expression":int(len(present)),
            "expression_source":str(path),
            "expression_features":int(len(expr_cols)),
        }
        if len(present)>=2:
            X=d.loc[present,expr_cols].apply(pd.to_numeric,errors="coerce").to_numpy(float)
            q=np.isfinite(X)
            col_ok=q.sum(axis=0)>=2
            X=X[:,col_ok]
            if X.shape[1]:
                means=np.nanmean(X,axis=0)
                Z=(X-means[None,:])**2
                vals=Z[np.isfinite(Z)]
                rec["within_expression_mse"]=float(vals.mean()) if len(vals) else np.nan
                rec["within_expression_feature_median_variance"]=float(np.nanmedian(np.nanvar(X,axis=0,ddof=0)))
            else:
                rec["within_expression_mse"]=np.nan
                rec["within_expression_feature_median_variance"]=np.nan
        else:
            rec["within_expression_mse"]=np.nan
            rec["within_expression_feature_median_variance"]=np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

def classify_units(mass,heterogeneity,cfg):
    """
    Descriptive quadrant classification. Cutoffs are within-landmark quantiles,
    not scientific gates.
    """
    m=np.asarray(mass,float)
    h=np.asarray(heterogeneity,float)
    out=np.full(len(m),"unresolved",dtype=object)
    mq=safe_quantile(m,cfg["large_mass_quantile"])
    hq=safe_quantile(h,cfg["high_heterogeneity_quantile"])
    if not np.isfinite(mq) or not np.isfinite(hq):return out,mq,hq
    for i,(mi,hi) in enumerate(zip(m,h)):
        if not np.isfinite(mi) or not np.isfinite(hi):continue
        large=mi>=mq; high=hi>=hq
        if large and not high:out[i]="large_coherent_candidate"
        elif large and high:out[i]="large_heterogeneous_candidate"
        elif (not large) and not high:out[i]="small_coherent_candidate"
        else:out[i]="small_heterogeneous_candidate"
    return out,mq,hq
