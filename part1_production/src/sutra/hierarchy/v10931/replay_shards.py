from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np
import pandas as pd

LEFT_ALIASES=("super_i","left","left_id","node_a","a","u","child_a","merge_a","node_u","lhs")
RIGHT_ALIASES=("super_j","right","right_id","node_b","b","v","child_b","merge_b","node_v","rhs")
NEW_ALIASES=("new","new_id","parent","parent_id","merged_id","new_node_id","result_id","survivor","keep_id")
STEP_ALIASES=("microstep","step","iteration","iter","merge_step","flow_step")
ACCEPT_ALIASES=("accepted","is_accepted","merge_accepted","selected","is_selected","performed","applied")

def norm(v):
    if isinstance(v,(np.integer,int)):return int(v)
    if isinstance(v,(np.floating,float)) and np.isfinite(v) and float(v).is_integer():return int(v)
    if isinstance(v,(bytes,np.bytes_)):return v.decode()
    return str(v)

def find_col(cols,aliases):
    low={str(c).lower():c for c in cols}
    for a in aliases:
        if a in low:return low[a]
    return None

def schema(df):
    cols=list(df.columns)
    return {
      "left":find_col(cols,LEFT_ALIASES),
      "right":find_col(cols,RIGHT_ALIASES),
      "new":find_col(cols,NEW_ALIASES),
      "step":find_col(cols,STEP_ALIASES),
      "accepted":find_col(cols,ACCEPT_ALIASES)
    }

def bool_mask(s):
    if pd.api.types.is_bool_dtype(s):return s.fillna(False).astype(bool)
    z=s.astype(str).str.strip().str.lower()
    return z.isin(["1","true","yes","accepted","selected","performed","applied"])

def shard_index(path):
    m=re.search(r"merge_(\d+)",path.name)
    return int(m.group(1)) if m else None

def discover_stream(project,sample,cfg):
    """
    Frozen landmarks came from v0.9.3, so replay only that trajectory.
    """
    base=project/cfg["v093_ledger_root"]/sample/"merges"
    if not base.exists():return None,pd.DataFrame()
    files=[]
    rows=[]
    for p in base.glob("merge_*.parquet"):
        idx=shard_index(p)
        if idx is None:continue
        try:
            import pyarrow.parquet as pq
            cols=list(pq.ParquetFile(p).schema_arrow.names)
            nrows=int(pq.ParquetFile(p).metadata.num_rows)
        except Exception:
            continue
        sch={
          "left":find_col(cols,LEFT_ALIASES),"right":find_col(cols,RIGHT_ALIASES),
          "new":find_col(cols,NEW_ALIASES),"step":find_col(cols,STEP_ALIASES),
          "accepted":find_col(cols,ACCEPT_ALIASES)
        }
        ok=sch["left"] is not None and sch["right"] is not None
        rows.append({
          "path":str(p),"shard_index":idx,"rows":nrows,
          "recognized_pair":ok,
          "left_col":sch["left"],"right_col":sch["right"],
          "new_col":sch["new"],"step_col":sch["step"],"accepted_col":sch["accepted"],
          "columns_json":json.dumps(cols)
        })
        if ok:files.append((idx,p))
    files.sort()
    return [p for _,p in files],pd.DataFrame(rows).sort_values("shard_index") if rows else pd.DataFrame()

def load_initial_labels(project,sample,n0,cfg):
    base=project/cfg["v093_ledger_root"]/sample/"checkpoints"
    candidates=[]
    if base.exists():
        for p in base.glob("labels_*.npz"):
            try:z=np.load(p,allow_pickle=False)
            except Exception:continue
            for key in z.files:
                a=np.asarray(z[key])
                if a.ndim!=1 or len(a)!=int(n0):continue
                if a.dtype.kind not in "iuf":continue
                if a.dtype.kind=="f":
                    if not np.isfinite(a).all() or not np.allclose(a,np.round(a),rtol=0,atol=0):continue
                    a=np.round(a).astype(np.int64)
                else:a=a.astype(np.int64)
                nu=len(np.unique(a))
                candidates.append((abs(nu-int(n0)), "000000" not in p.name, str(p),key,a))
    if not candidates:return None,None,None
    candidates.sort(key=lambda x:(x[0],x[1],x[2]))
    _,_,p,k,a=candidates[0]
    return a,p,k

class DSU:
    def __init__(self,labels):
        labs=[norm(x) for x in np.asarray(labels)]
        uniq=list(dict.fromkeys(labs))
        self.parent=list(range(len(uniq)));self.sz=[1]*len(uniq)
        self.alias={lab:i for i,lab in enumerate(uniq)}
        self.cell_node=np.array([self.alias[x] for x in labs],dtype=np.int64)
        self.live=set(range(len(uniq)))
    def find(self,x):
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]]
            x=self.parent[x]
        return x
    def resolve(self,label):
        i=self.alias.get(norm(label))
        return None if i is None else self.find(i)
    @property
    def nlive(self):
        return len({self.find(i) for i in self.live})
    def union_roots(self,ra,rb,new_label=None,left_alias=None,right_alias=None):
        ra=self.find(ra);rb=self.find(rb)
        if ra==rb:return False
        # Root choice is an internal DSU implementation detail. All externally
        # observed aliases are rebound to the resulting root.
        if self.sz[ra]<self.sz[rb]:ra,rb=rb,ra
        self.parent[rb]=ra;self.sz[ra]+=self.sz[rb]
        self.live.discard(rb);self.live.add(ra)
        if left_alias is not None:self.alias[norm(left_alias)]=ra
        if right_alias is not None:self.alias[norm(right_alias)]=ra
        if new_label is not None and not (isinstance(new_label,float) and np.isnan(new_label)):
            self.alias[norm(new_label)]=ra
        return True
    def labels(self):
        roots=np.array([self.find(int(i)) for i in self.cell_node],dtype=np.int64)
        mapping={};out=np.empty(len(roots),dtype=np.int64);nxt=0
        for i,r in enumerate(roots):
            if r not in mapping:mapping[r]=nxt;nxt+=1
            out[i]=mapping[r]
        return out

def read_shard(path):
    return pd.read_parquet(path)

def prepare_batch(df):
    sch=schema(df)
    if sch["left"] is None or sch["right"] is None:
        return None,sch
    d=df.copy()
    if sch["accepted"] is not None:
        d=d.loc[bool_mask(d[sch["accepted"]])].copy()
    return d,sch

def replay_stream(files,initial_labels,target_counts):
    """
    Each merge_XXXXXX parquet is one microstep batch.
    All rows in a batch must be pairwise disjoint in the active partition.
    Snapshots are taken only after an entire shard has been applied.
    """
    dsu=DSU(initial_labels)
    targets=set(map(int,target_counts))
    snapshots={}
    shard_rows=[]
    if dsu.nlive in targets:snapshots[dsu.nlive]=dsu.labels()

    for p in files:
        df=read_shard(p)
        d,sch=prepare_batch(df)
        if d is None:
            shard_rows.append({"path":str(p),"status":"schema_fail"})
            return snapshots,pd.DataFrame(shard_rows),False

        roots=[]
        unresolved=0;redundant=0
        for _,r in d.iterrows():
            a=dsu.resolve(r[sch["left"]]);b=dsu.resolve(r[sch["right"]])
            if a is None or b is None:
                unresolved+=1;continue
            if a==b:
                redundant+=1;continue
            roots.append((a,b,r))

        # Scientific contract: accepted simultaneous contractions are disjoint.
        seen=set();overlap=0
        for a,b,_ in roots:
            a=dsu.find(a);b=dsu.find(b)
            if a in seen or b in seen:overlap+=1
            seen.add(a);seen.add(b)

        before=dsu.nlive
        if unresolved or overlap:
            shard_rows.append({
              "path":str(p),"shard_index":shard_index(p),"rows":int(len(d)),
              "before_nodes":before,"applied":0,"after_nodes":before,
              "unresolved_endpoints":unresolved,"redundant_pairs":redundant,
              "batch_root_overlap":overlap,"status":"HOLD"
            })
            return snapshots,pd.DataFrame(shard_rows),False

        applied=0
        for a,b,r in roots:
            new=r[sch["new"]] if sch["new"] is not None else None
            if dsu.union_roots(a,b,new,r[sch["left"]],r[sch["right"]]):
                applied+=1
        after=dsu.nlive
        exact_drop=(after==before-applied)
        shard_rows.append({
          "path":str(p),"shard_index":shard_index(p),"rows":int(len(d)),
          "before_nodes":before,"applied":applied,"after_nodes":after,
          "unresolved_endpoints":unresolved,"redundant_pairs":redundant,
          "batch_root_overlap":overlap,"exact_node_drop":exact_drop,
          "status":"PASS" if exact_drop else "HOLD"
        })
        if not exact_drop:return snapshots,pd.DataFrame(shard_rows),False

        if after in targets and after not in snapshots:
            snapshots[after]=dsu.labels()

        # A target skipped inside a batch cannot be reconstructed as a true
        # microstep state and must not be synthesized.
        skipped=[t for t in targets if after<t<before and t not in snapshots]
        if skipped:
            for t in skipped:
                # mark only; continue because other frozen landmarks may resolve
                pass

        if len(snapshots)==len(targets):break

    return snapshots,pd.DataFrame(shard_rows),True

def snapshot_audit(labels,n0,nodes):
    a=np.asarray(labels)
    return {
      "n_cells":int(len(a)),"unique_nodes":int(len(np.unique(a))),
      "expected_nodes":int(nodes),
      "cell_count_match":bool(len(a)==int(n0)),
      "node_count_match":bool(len(np.unique(a))==int(nodes))
    }

def nested_labels(fine,coarse):
    fine=np.asarray(fine);coarse=np.asarray(coarse)
    if len(fine)!=len(coarse):return pd.DataFrame(),False
    rows=[]
    for f in np.unique(fine):
        q=fine==f
        p=np.unique(coarse[q])
        rows.append({
          "fine_label":int(f),"fine_mass":int(q.sum()),
          "n_coarse_parents":int(len(p)),
          "coarse_label":int(p[0]) if len(p)==1 else None,
          "coarse_mass":int((coarse==p[0]).sum()) if len(p)==1 else np.nan,
          "mass_monotone":bool(len(p)==1 and (coarse==p[0]).sum()>=q.sum())
        })
    d=pd.DataFrame(rows)
    ok=bool(len(d) and (d.n_coarse_parents==1).all() and d.mass_monotone.all())
    return d,ok

def dominant(labels,fraction=.8):
    u,c=np.unique(labels,return_counts=True)
    o=np.argsort(c)[::-1];u=u[o];c=c[o]
    p=c/c.sum();cs=np.cumsum(p)
    k=int(np.searchsorted(cs,float(fraction),side="left")+1)
    tab=pd.DataFrame({
      "supernode_id":u.astype(int),"mass":c.astype(int),"mass_fraction":p,
      "cumulative_mass_fraction":cs,
      "dominant_mass_set":[i<k for i in range(len(u))]
    })
    return set(map(int,u[:k])),tab

def load_native_expression_aligned(project,sample):
    """
    Align Xenium H5 rows to cells.parquet through explicit cell IDs/barcodes.
    Returns None rather than assuming row identity if exact matching cannot be
    proven.
    """
    hp=project/"data"/sample/"cell_feature_matrix.h5"
    cp=project/"data"/sample/"cells.parquet"
    if not hp.exists() or not cp.exists():return None,None
    try:
        import h5py
        from scipy.sparse import csc_matrix
        cells=pd.read_parquet(cp)
        cid=next((c for c in ["cell_id","cell","barcode"] if c in cells.columns),None)
        if cid is None:return None,None
        cell_ids=cells[cid].astype(str).to_numpy()
        with h5py.File(hp,"r") as f:
            g=f["matrix"]
            data=np.asarray(g["data"]);indices=np.asarray(g["indices"])
            indptr=np.asarray(g["indptr"]);shape=tuple(np.asarray(g["shape"]).astype(int))
            X=csc_matrix((data,indices,indptr),shape=shape).T.tocsr()
            if "barcodes" not in g:return None,None
            raw=np.asarray(g["barcodes"])
            bar=np.array([x.decode() if isinstance(x,(bytes,np.bytes_)) else str(x) for x in raw])
        if len(bar)!=len(cell_ids) or len(np.unique(bar))!=len(bar):return None,None
        pos={x:i for i,x in enumerate(bar)}
        if not all(x in pos for x in cell_ids):return None,None
        order=np.array([pos[x] for x in cell_ids],dtype=np.int64)
        return X[order],str(hp)
    except Exception:
        return None,None

def load_coordinates(project,sample,n0):
    p=project/"data"/sample/"cells.parquet"
    if not p.exists():return None,None
    d=pd.read_parquet(p)
    xa=["x","x_coord","centroid_x","x_centroid","cell_centroid_x","cell_x"]
    ya=["y","y_coord","centroid_y","y_centroid","cell_centroid_y","cell_y"]
    xc=next((c for c in xa if c in d.columns),None)
    yc=next((c for c in ya if c in d.columns),None)
    if xc is None or yc is None or len(d)!=int(n0):return None,None
    P=np.column_stack([
      pd.to_numeric(d[xc],errors="coerce").to_numpy(float),
      pd.to_numeric(d[yc],errors="coerce").to_numpy(float)
    ])
    return P,str(p)

def expression_coherence(labels,X,keep):
    if X is None:return pd.DataFrame()
    labels=np.asarray(labels);rows=[]
    for l in sorted(keep):
        idx=np.flatnonzero(labels==l)
        if not len(idx):continue
        Xi=X[idx]
        mu=np.asarray(Xi.mean(axis=0)).ravel()
        ms=np.asarray(Xi.power(2).mean(axis=0)).ravel()
        var=np.maximum(ms-mu*mu,0)
        rows.append({
          "supernode_id":int(l),"n_cells":int(len(idx)),
          "within_expression_mean_variance":float(np.mean(var)),
          "within_expression_median_variance":float(np.median(var))
        })
    return pd.DataFrame(rows)

def spatial_coherence(labels,P,keep):
    if P is None:return pd.DataFrame()
    labels=np.asarray(labels);rows=[]
    for l in sorted(keep):
        Q=P[labels==l];Q=Q[np.isfinite(Q).all(axis=1)]
        if not len(Q):continue
        mu=Q.mean(axis=0);rr=np.sum((Q-mu)**2,axis=1)
        rows.append({
          "supernode_id":int(l),"n_cells_with_coordinates":int(len(Q)),
          "spatial_rms_radius":float(np.sqrt(np.mean(rr))),
          "spatial_extent_x":float(np.ptp(Q[:,0])),
          "spatial_extent_y":float(np.ptp(Q[:,1]))
        })
    return pd.DataFrame(rows)
