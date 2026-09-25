from __future__ import annotations
import ast, json, re
from pathlib import Path
import numpy as np
import pandas as pd

LEFT_ALIASES = (
    "left","left_id","node_a","a","u","source_a","child_a","merge_a",
    "node_u","first","lhs","src_a","absorber"
)
RIGHT_ALIASES = (
    "right","right_id","node_b","b","v","source_b","child_b","merge_b",
    "node_v","second","rhs","src_b","absorbed"
)
NEW_ALIASES = (
    "new","new_id","parent","parent_id","merged_id","new_node_id",
    "result","result_id","survivor","keep_id","target_id"
)
STEP_ALIASES = (
    "microstep","step","iteration","iter","merge_step","flow_step","event_index","index"
)
ACCEPT_ALIASES = (
    "accepted","is_accepted","merge_accepted","selected","is_selected","performed","applied"
)

def normalize_scalar(v):
    if isinstance(v,(np.integer,int)):
        return int(v)
    if isinstance(v,(np.floating,float)):
        if np.isfinite(v) and float(v).is_integer():
            return int(v)
        return str(v)
    if isinstance(v,(bytes,np.bytes_)):
        return v.decode()
    return str(v)

def bool_mask(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    z=s.astype(str).str.strip().str.lower()
    return z.isin(["1","true","yes","accepted","selected","performed","applied"])

def find_column(cols,aliases):
    low={str(c).lower():c for c in cols}
    for a in aliases:
        if a in low:return low[a]
    return None

def inspect_merge_schema(df):
    cols=list(df.columns)
    lc=find_column(cols,LEFT_ALIASES)
    rc=find_column(cols,RIGHT_ALIASES)
    nc=find_column(cols,NEW_ALIASES)
    sc=find_column(cols,STEP_ALIASES)
    ac=find_column(cols,ACCEPT_ALIASES)
    return {"left":lc,"right":rc,"new":nc,"step":sc,"accepted":ac}

def read_table(path):
    suf=path.suffix.lower()
    try:
        if suf==".parquet":
            return pd.read_parquet(path)
        if suf==".csv":
            return pd.read_csv(path)
        if str(path).lower().endswith(".csv.gz"):
            return pd.read_csv(path,compression="gzip")
        if suf in {".jsonl",".ndjson"}:
            return pd.read_json(path,lines=True)
        if suf==".json":
            x=json.loads(path.read_text())
            if isinstance(x,list):return pd.DataFrame(x)
            if isinstance(x,dict):
                for k,v in x.items():
                    if isinstance(v,list) and v and isinstance(v[0],dict):
                        return pd.DataFrame(v)
    except Exception:
        return None
    return None

def discover_merge_tables(project,sample,cfg):
    rows=[]
    paths=[]
    roots=[]
    for rel in cfg["merge_search_roots"]:
        p=project/rel
        if p.exists():roots.append(p)
    sample_low=sample.lower()
    for rr in roots:
        for p in rr.rglob("*"):
            if not p.is_file():continue
            low=str(p).lower()
            if sample_low not in low and p.parent.name.lower()!=sample_low:
                # sample-agnostic ledgers are still inspected only when the filename
                # itself strongly suggests merge ancestry.
                if not any(k in p.name.lower() for k in ["merge","ancestry","contraction"]):
                    continue
            if not (p.suffix.lower() in {".parquet",".csv",".json",".jsonl",".ndjson"} or str(p).lower().endswith(".csv.gz")):
                continue
            name_hint=sum(k in p.name.lower() for k in ["merge","ancestry","contraction","ledger","flow"])
            if name_hint==0:continue
            d=read_table(p)
            if d is None or len(d)==0:continue
            sch=inspect_merge_schema(d)
            recognized=sch["left"] is not None and sch["right"] is not None
            rows.append({
                "path":str(p),"rows":int(len(d)),
                "name_hint_score":int(name_hint),
                "recognized_pair":bool(recognized),
                "left_col":sch["left"],"right_col":sch["right"],
                "new_col":sch["new"],"step_col":sch["step"],
                "accepted_col":sch["accepted"],
                "columns_json":json.dumps(list(map(str,d.columns)))
            })
            if recognized:paths.append(p)
    return pd.DataFrame(rows),paths

def load_initial_labels(project,sample,n0,cfg):
    candidates=[]
    for rel in cfg["initial_label_search_roots"]:
        rr=project/rel
        if not rr.exists():continue
        for p in rr.rglob("*.npz"):
            if sample.lower() not in str(p).lower():continue
            if "label" not in p.name.lower():continue
            try:z=np.load(p,allow_pickle=False)
            except Exception:continue
            for k in z.files:
                a=np.asarray(z[k])
                if a.ndim!=1 or len(a)!=int(n0):continue
                if a.dtype.kind not in "iuf":continue
                if a.dtype.kind=="f":
                    if not np.isfinite(a).all() or not np.allclose(a,np.round(a),rtol=0,atol=0):
                        continue
                    a=np.round(a).astype(np.int64)
                else:a=a.astype(np.int64)
                nu=len(np.unique(a))
                # Prefer a true Level-0 identity partition.
                candidates.append((abs(nu-int(n0)),p,k,a,nu))
    if not candidates:return None,None,None
    candidates.sort(key=lambda x:(x[0], "000000" not in x[1].name, str(x[1])))
    _,p,k,a,nu=candidates[0]
    return a,str(p),str(k)

class DSU:
    def __init__(self, initial_labels):
        labs=[normalize_scalar(x) for x in np.asarray(initial_labels)]
        uniq=list(dict.fromkeys(labs))
        self.parent=list(range(len(uniq)))
        self.sz=[1]*len(uniq)
        self.label_to_index={lab:i for i,lab in enumerate(uniq)}
        self.cell_nodes=np.array([self.label_to_index[x] for x in labs],dtype=np.int64)
        self.live=set(range(len(uniq)))
    def find(self,x):
        p=self.parent
        while p[x]!=x:
            p[x]=p[p[x]];x=p[x]
        return x
    def resolve_label(self,label):
        label=normalize_scalar(label)
        i=self.label_to_index.get(label,None)
        return None if i is None else self.find(i)
    def union(self,a,b,new_label=None):
        ra=self.resolve_label(a);rb=self.resolve_label(b)
        if ra is None or rb is None or ra==rb:return False
        if self.sz[ra]<self.sz[rb]:ra,rb=rb,ra
        self.parent[rb]=ra
        self.sz[ra]+=self.sz[rb]
        self.live.discard(rb);self.live.add(ra)
        if new_label is not None and not (isinstance(new_label,float) and np.isnan(new_label)):
            self.label_to_index[normalize_scalar(new_label)]=ra
        # keep the original input aliases pointing to the merged root too.
        self.label_to_index[normalize_scalar(a)]=ra
        self.label_to_index[normalize_scalar(b)]=ra
        return True
    @property
    def live_count(self):
        return len({self.find(x) for x in self.live})
    def labels(self):
        roots=np.array([self.find(int(x)) for x in self.cell_nodes],dtype=np.int64)
        uniq={}
        out=np.empty(len(roots),dtype=np.int64)
        nxt=0
        for i,r in enumerate(roots):
            if r not in uniq:
                uniq[r]=nxt;nxt+=1
            out[i]=uniq[r]
        return out

def normalize_merge_rows(df,sch):
    d=df.copy()
    if sch["accepted"] is not None:
        d=d.loc[bool_mask(d[sch["accepted"]])].copy()
    if sch["step"] is not None:
        d["_step_numeric"]=pd.to_numeric(d[sch["step"]],errors="coerce")
        if d["_step_numeric"].notna().any():
            d=d.sort_values("_step_numeric",kind="stable")
    return d

def replay_merge_table(df,initial_labels,target_counts):
    sch=inspect_merge_schema(df)
    if sch["left"] is None or sch["right"] is None:
        return None
    d=normalize_merge_rows(df,sch)
    dsu=DSU(initial_labels)
    targets=set(map(int,target_counts))
    snapshots={}
    stats=[]
    initial_live=dsu.live_count
    if initial_live in targets:snapshots[initial_live]=dsu.labels()

    applied=0;unresolved=0;redundant=0
    for row_index,(_,r) in enumerate(d.iterrows()):
        left=r[sch["left"]];right=r[sch["right"]]
        new=r[sch["new"]] if sch["new"] is not None else left
        ra=dsu.resolve_label(left);rb=dsu.resolve_label(right)
        if ra is None or rb is None:
            unresolved+=1
            continue
        if ra==rb:
            redundant+=1
            if sch["new"] is not None:
                dsu.label_to_index[normalize_scalar(new)]=ra
            continue
        before=dsu.live_count
        ok=dsu.union(left,right,new)
        after=dsu.live_count
        if ok:
            applied+=1
            if after != before-1:
                raise RuntimeError("DSU live-node invariant violated")
            if after in targets and after not in snapshots:
                snapshots[after]=dsu.labels()
        if len(snapshots)==len(targets):
            break
    return {
        "snapshots":snapshots,
        "initial_live":initial_live,
        "final_live":dsu.live_count,
        "applied_merges":applied,
        "unresolved_rows":unresolved,
        "redundant_rows":redundant,
        "rows_considered":int(len(d)),
        "schema":sch
    }

def audit_snapshot(labels,n0,expected_nodes):
    a=np.asarray(labels)
    return {
        "n_cells":int(len(a)),
        "unique_nodes":int(len(np.unique(a))),
        "expected_nodes":int(expected_nodes),
        "cell_count_match":bool(len(a)==int(n0)),
        "node_count_match":bool(len(np.unique(a))==int(expected_nodes))
    }

def nested_labels(fine,coarse):
    fine=np.asarray(fine);coarse=np.asarray(coarse)
    if len(fine)!=len(coarse):return pd.DataFrame(),False
    rows=[]
    for f in np.unique(fine):
        q=fine==f
        parents=np.unique(coarse[q])
        rows.append({
            "fine_label":int(f),"fine_mass":int(q.sum()),
            "n_coarse_parents":int(len(parents)),
            "coarse_label":int(parents[0]) if len(parents)==1 else None,
            "coarse_mass":int((coarse==parents[0]).sum()) if len(parents)==1 else np.nan,
            "mass_monotone":bool(len(parents)==1 and (coarse==parents[0]).sum()>=q.sum())
        })
    d=pd.DataFrame(rows)
    ok=bool(len(d) and (d.n_coarse_parents==1).all() and d.mass_monotone.all())
    return d,ok

def dominant_labels(labels,mass_fraction=.8):
    u,c=np.unique(labels,return_counts=True)
    order=np.argsort(c)[::-1]
    u=u[order];c=c[order]
    frac=c/c.sum();cum=np.cumsum(frac)
    k=int(np.searchsorted(cum,float(mass_fraction),side="left")+1)
    return set(map(int,u[:k])),pd.DataFrame({
        "label":u.astype(int),"mass":c.astype(int),"mass_fraction":frac,
        "cumulative_mass_fraction":cum,
        "dominant_K80":[i<k for i in range(len(u))]
    })

def load_xenium_expression(project,sample):
    p=project/"data"/sample/"cell_feature_matrix.h5"
    if not p.exists():return None,None,None
    try:
        import h5py
        from scipy.sparse import csc_matrix
        with h5py.File(p,"r") as f:
            g=f["matrix"]
            data=np.asarray(g["data"]);indices=np.asarray(g["indices"])
            indptr=np.asarray(g["indptr"]);shape=tuple(np.asarray(g["shape"]).astype(int))
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

def load_xenium_coordinates(project,sample,n0):
    p=project/"data"/sample/"cells.parquet"
    if not p.exists():return None,None
    try:d=pd.read_parquet(p)
    except Exception:return None,None
    xaliases=["x","x_coord","centroid_x","x_centroid","cell_centroid_x","cell_x"]
    yaliases=["y","y_coord","centroid_y","y_centroid","cell_centroid_y","cell_y"]
    xc=next((c for c in xaliases if c in d.columns),None)
    yc=next((c for c in yaliases if c in d.columns),None)
    if xc is None or yc is None:return None,None
    x=pd.to_numeric(d[xc],errors="coerce").to_numpy(float)
    y=pd.to_numeric(d[yc],errors="coerce").to_numpy(float)
    if len(x)!=int(n0):return None,None
    return np.column_stack([x,y]),str(p)

def expression_coherence(labels,X,dominant):
    if X is None:return pd.DataFrame()
    labels=np.asarray(labels);rows=[]
    for l in sorted(dominant):
        idx=np.where(labels==l)[0]
        if len(idx)==0:continue
        Xi=X[idx]
        mean=np.asarray(Xi.mean(axis=0)).ravel()
        mean_sq=np.asarray(Xi.power(2).mean(axis=0)).ravel()
        var=np.maximum(mean_sq-mean*mean,0)
        rows.append({
            "supernode_id":int(l),"n_cells":int(len(idx)),
            "within_expression_mean_variance":float(np.mean(var)),
            "within_expression_median_variance":float(np.median(var))
        })
    return pd.DataFrame(rows)

def spatial_coherence(labels,coords,dominant):
    if coords is None:return pd.DataFrame()
    labels=np.asarray(labels);rows=[]
    for l in sorted(dominant):
        P=coords[labels==l]
        P=P[np.isfinite(P).all(axis=1)]
        if len(P)==0:continue
        mu=P.mean(axis=0);rr=np.sum((P-mu)**2,axis=1)
        cov=np.cov(P,rowvar=False,bias=True) if len(P)>=2 else np.zeros((2,2))
        ev=np.linalg.eigvalsh(cov)
        rows.append({
            "supernode_id":int(l),"n_cells_with_coordinates":int(len(P)),
            "spatial_rms_radius":float(np.sqrt(np.mean(rr))),
            "spatial_extent_x":float(np.ptp(P[:,0])),
            "spatial_extent_y":float(np.ptp(P[:,1])),
            "spatial_anisotropy_ratio":float(ev[-1]/max(ev[0],1e-12)) if len(ev)==2 else np.nan
        })
    return pd.DataFrame(rows)
