from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

def derive_level0_count(project,sample):
    # Prefer native cells.parquet because it defines the cell universe directly.
    p=project/"data"/sample/"cells.parquet"
    if p.exists():
        try:
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(p).metadata.num_rows),str(p),"cells.parquet_rows"
        except Exception:
            pass
    # Fall back to 10x/Xenium feature matrix shape.
    h=project/"data"/sample/"cell_feature_matrix.h5"
    if h.exists():
        try:
            import h5py
            with h5py.File(h,"r") as f:
                if "matrix/shape" in f:
                    shape=np.asarray(f["matrix/shape"])
                    # 10x HDF5 shape is [features,cells]
                    return int(shape[-1]),str(h),"matrix_shape"
        except Exception:
            pass
    return None,None,None

def normalize_labels(a):
    a=np.asarray(a)
    if a.ndim!=1:return None
    if a.dtype.kind not in "iu":
        # exact integer-valued floating labels are acceptable
        if a.dtype.kind=="f" and np.isfinite(a).all() and np.allclose(a,np.round(a),rtol=0,atol=0):
            a=np.round(a).astype(np.int64)
        else:return None
    return a.astype(np.int64,copy=False)

def inspect_npz_partition_candidates(path,n0,expected_nodes):
    rows=[]; payloads={}
    try:
        z=np.load(path,allow_pickle=False)
    except Exception:
        return rows,payloads
    for key in z.files:
        try:a=z[key]
        except Exception:continue
        lab=normalize_labels(a)
        if lab is None or len(lab)!=int(n0):continue
        u=np.unique(lab)
        rec={
            "path":str(path),"kind":"npz_label_vector","array_key":str(key),
            "length":int(len(lab)),"unique_labels":int(len(u)),
            "expected_nodes":int(expected_nodes),
            "exact_node_match":bool(len(u)==int(expected_nodes)),
        }
        rows.append(rec)
        if rec["exact_node_match"]:
            payloads[(str(path),str(key))]=lab
    return rows,payloads

def inspect_parquet_partition_candidate(path,n0,expected_nodes):
    try:
        import pyarrow.parquet as pq
        pf=pq.ParquetFile(path)
        cols=list(pf.schema_arrow.names)
    except Exception:
        return [],{}
    cellc=next((c for c in ["cell_id","cell","barcode","cell_index","level0_id"] if c in cols),None)
    labc=next((c for c in ["supernode_id","node_id","label","cluster_id","parent_id"] if c in cols),None)
    if cellc is None or labc is None:return [],{}
    try:
        d=pd.read_parquet(path,columns=[cellc,labc])
    except Exception:return [],{}
    if len(d)!=int(n0):return [],{}
    if d[cellc].astype(str).nunique()!=int(n0):return [],{}
    nlab=int(d[labc].nunique(dropna=False))
    rec={
        "path":str(path),"kind":"parquet_cell_assignment","array_key":labc,
        "length":int(len(d)),"unique_labels":nlab,
        "expected_nodes":int(expected_nodes),
        "exact_node_match":bool(nlab==int(expected_nodes)),
    }
    payload={}
    if rec["exact_node_match"]:
        # preserve row order only after sorting by explicit cell index if numeric;
        # otherwise payload remains dataframe-backed and caller joins by cell id.
        payload[(str(path),labc)]=d
    return [rec],payload

def scan_assignment_candidates(project,sample,n0,expected_nodes,cfg):
    roots=[]
    for rel in cfg["ledger_roots"]:
        p=project/rel/sample
        if p.exists():roots.append(p)
    rows=[];payloads={}
    for rr in roots:
        for p in rr.rglob("*"):
            if not p.is_file():continue
            low=p.name.lower()
            # assignment/checkpoint sources only; do not treat node-statistic tables as labels.
            if p.suffix.lower()==".npz":
                r,q=inspect_npz_partition_candidates(p,n0,expected_nodes)
            elif p.suffix.lower()==".parquet":
                r,q=inspect_parquet_partition_candidate(p,n0,expected_nodes)
            else:
                continue
            rows.extend(r);payloads.update(q)
    return pd.DataFrame(rows),payloads

def choose_exact_candidate(audit):
    if audit is None or len(audit)==0:return None
    q=audit[audit.exact_node_match.astype(bool)]
    # Exact uniqueness is required. Duplicate physical copies with identical
    # basename/key are handled only if they represent the same source path.
    if len(q)!=1:return None
    return q.iloc[0].to_dict()

def membership_map_from_label_vector(labels):
    labels=np.asarray(labels)
    out={}
    for i,l in enumerate(labels):
        out.setdefault(str(int(l)),set()).add(int(i))
    return out

def membership_map_from_assignment_df(df):
    cols=list(df.columns)
    cellc=next((c for c in ["cell_id","cell","barcode","cell_index","level0_id"] if c in cols),None)
    labc=next((c for c in ["supernode_id","node_id","label","cluster_id","parent_id"] if c in cols),None)
    if cellc is None or labc is None:return {}
    out={}
    for c,l in zip(df[cellc].astype(str),df[labc].astype(str)):
        out.setdefault(l,set()).add(c)
    return out

def partition_invariants(m):
    if not m:return {"nodes":0,"total":0,"unique":0,"duplicates":0,"disjoint":False}
    seen={}
    total=0
    for S in m.values():
        total+=len(S)
        for x in S:seen[x]=seen.get(x,0)+1
    dup=sum(v-1 for v in seen.values() if v>1)
    return {
        "nodes":int(len(m)),"total":int(total),"unique":int(len(seen)),
        "duplicates":int(dup),"disjoint":bool(dup==0)
    }

def compare_nested(child,parent):
    rows=[]
    if not child or not parent:return pd.DataFrame(),False
    for cid,C in child.items():
        cand=[(pid,P) for pid,P in parent.items() if C.issubset(P)]
        rows.append({
            "child_supernode_id":cid,"child_mass":len(C),
            "n_exact_parents":len(cand),
            "parent_supernode_id":cand[0][0] if len(cand)==1 else None,
            "parent_mass":len(cand[0][1]) if len(cand)==1 else np.nan,
            "mass_monotone":bool(len(cand)==1 and len(cand[0][1])>=len(C))
        })
    d=pd.DataFrame(rows)
    ok=bool(len(d) and (d.n_exact_parents==1).all() and d.mass_monotone.all())
    return d,ok

def load_xenium_expression(project,sample):
    """
    Returns sparse cells x genes matrix plus feature names when native H5 exists.
    """
    p=project/"data"/sample/"cell_feature_matrix.h5"
    if not p.exists():return None,None,None
    try:
        import h5py
        from scipy.sparse import csc_matrix
        with h5py.File(p,"r") as f:
            g=f["matrix"]
            data=np.asarray(g["data"])
            indices=np.asarray(g["indices"])
            indptr=np.asarray(g["indptr"])
            shape=tuple(np.asarray(g["shape"]).astype(int))
            X=csc_matrix((data,indices,indptr),shape=shape).T.tocsr()
            names=None
            for key in ["features/name","features/id"]:
                if key in g:
                    raw=np.asarray(g[key])
                    names=np.array([x.decode() if isinstance(x,(bytes,np.bytes_)) else str(x) for x in raw])
                    break
        return X,names,str(p)
    except Exception:
        return None,None,None

def expression_coherence(label_vector,X):
    if label_vector is None or X is None:return pd.DataFrame()
    labs=np.asarray(label_vector)
    rows=[]
    for l in np.unique(labs):
        idx=np.where(labs==l)[0]
        if len(idx)<2:
            rows.append({"supernode_id":int(l),"n_cells":int(len(idx)),
                         "within_expression_mse":0.0})
            continue
        Xi=X[idx]
        mean=np.asarray(Xi.mean(axis=0)).ravel()
        mean_sq=np.asarray(Xi.power(2).mean(axis=0)).ravel()
        var=np.maximum(mean_sq-mean*mean,0)
        rows.append({
            "supernode_id":int(l),"n_cells":int(len(idx)),
            "within_expression_mse":float(np.mean(var)),
            "within_expression_median_variance":float(np.median(var))
        })
    return pd.DataFrame(rows)

def load_coordinates(project,sample,n0):
    p=project/"data"/sample/"cells.parquet"
    if not p.exists():return None,None
    try:d=pd.read_parquet(p)
    except Exception:return None,None
    xc=next((c for c in ["x","x_coord","centroid_x","cell_centroid_x"] if c in d.columns),None)
    yc=next((c for c in ["y","y_coord","centroid_y","cell_centroid_y"] if c in d.columns),None)
    if xc is None or yc is None:return None,None
    x=pd.to_numeric(d[xc],errors="coerce").to_numpy(float)
    y=pd.to_numeric(d[yc],errors="coerce").to_numpy(float)
    if len(x)!=int(n0):return None,None
    return np.column_stack([x,y]),str(p)

def spatial_coherence(label_vector,coords):
    if label_vector is None or coords is None:return pd.DataFrame()
    labs=np.asarray(label_vector); rows=[]
    for l in np.unique(labs):
        q=(labs==l)
        P=coords[q]
        good=np.isfinite(P).all(axis=1);P=P[good]
        if len(P)==0:continue
        mu=P.mean(axis=0)
        rr=np.sum((P-mu)**2,axis=1)
        rows.append({
            "supernode_id":int(l),"n_cells_with_coordinates":int(len(P)),
            "spatial_rms_radius":float(np.sqrt(np.mean(rr))),
            "spatial_extent_x":float(P[:,0].max()-P[:,0].min()),
            "spatial_extent_y":float(P[:,1].max()-P[:,1].min())
        })
    return pd.DataFrame(rows)
