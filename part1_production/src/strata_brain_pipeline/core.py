
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

from strata_hierarchy.v10931.replay_shards import load_native_expression_aligned, load_coordinates

def ensure_dir(p:Path):
    p.mkdir(parents=True,exist_ok=True)
    return p

def sha256_file(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b=f.read(1<<20)
            if not b: break
            h.update(b)
    return h.hexdigest()

def find_required_certificate(project:Path,stage:str):
    base=project/"results"/stage
    if not base.exists():
        raise FileNotFoundError(f"Required result stage missing: {base}")
    hits=sorted(base.rglob("*certificate*.json"))
    if not hits:
        raise FileNotFoundError(f"No certificate found under {base}")
    return hits[0]

def validate_inputs(project:Path,cfg:dict):
    rep={"status":"PASS","samples":{},"required_stages":{},"scientific_contract":cfg["scientific_contract"]}
    for key,stage in cfg["required_results"].items():
        cert=find_required_certificate(project,stage)
        rep["required_stages"][key]={"stage":stage,"certificate":str(cert),"sha256":sha256_file(cert)}
    for s in cfg["samples"]:
        sample=s["id"]
        needed=[
            project/"data"/sample/"cell_feature_matrix.h5",
            project/"data"/sample/"cells.parquet",
            project/"results"/cfg["required_results"]["final_landscape"]/sample/"final_supported_pareto_landmarks.csv",
            project/"results"/cfg["required_results"]["merger_atlas"]/sample/"landmark_merger_structure.csv",
            project/"results"/cfg["required_results"]["merger_atlas"]/sample/"landmark_partition_audit.csv",
        ]
        missing=[str(x) for x in needed if not x.exists()]
        st="PASS" if not missing else "HOLD"
        if missing: rep["status"]="HOLD"
        rep["samples"][sample]={"status":st,"missing":missing}
    return rep

def extract_gene_names(h5_path:Path):
    import h5py
    with h5py.File(h5_path,"r") as f:
        fg=f["matrix"]["features"]
        for key in ["name","gene_names","id"]:
            if key in fg:
                raw=np.asarray(fg[key])
                return [x.decode() if isinstance(x,(bytes,np.bytes_)) else str(x) for x in raw]
    raise RuntimeError(f"{h5_path}: no feature names found")

def normalize_expression(X,mode):
    from scipy import sparse
    if mode!="log1p_library_size_10000":
        raise ValueError(mode)
    X=X.tocsr().astype(float)
    lib=np.asarray(X.sum(axis=1)).ravel()
    scale=np.divide(10000.0,lib,out=np.zeros_like(lib,dtype=float),where=lib>0)
    Y=sparse.diags(scale)@X
    Y.data=np.log1p(Y.data)
    return Y

def factorize_labels(labels):
    codes,uniq=pd.factorize(pd.Series(labels),sort=False)
    if np.any(codes<0): raise RuntimeError("Missing hierarchy labels")
    return codes.astype(np.int64),np.asarray(uniq)

def component_means(X,labels):
    from scipy import sparse
    codes,uniq=factorize_labels(labels)
    ncomp=len(uniq); n=len(codes)
    masses=np.bincount(codes,minlength=ncomp).astype(np.int64)
    vals=1.0/masses[codes]
    W=sparse.csr_matrix((vals,(codes,np.arange(n))),shape=(ncomp,n))
    return (W@X).tocsr(),masses,codes,uniq

def component_centroids(P,codes,ncomp):
    m=np.bincount(codes,minlength=ncomp)
    x=np.bincount(codes,weights=P[:,0],minlength=ncomp)
    y=np.bincount(codes,weights=P[:,1],minlength=ncomp)
    return np.column_stack([x/np.maximum(m,1),y/np.maximum(m,1)])

def parent_ids(entry_codes,center_labels):
    cc,_=pd.factorize(pd.Series(center_labels),sort=False)
    ncomp=int(entry_codes.max())+1
    parent=np.full(ncomp,-1,dtype=np.int64)
    for k in range(ncomp):
        u=np.unique(cc[entry_codes==k])
        if len(u)!=1: raise RuntimeError("Entry component maps to multiple center parents")
        parent[k]=int(u[0])
    return parent

def sample_actual_pairs(parent,max_pairs_per_parent,max_total,rng):
    groups={}
    for i,p in enumerate(parent): groups.setdefault(int(p),[]).append(i)
    pairs=[]
    for ch in groups.values():
        k=len(ch)
        if k<2: continue
        possible=k*(k-1)//2
        cap=min(max_pairs_per_parent,possible)
        if possible<=cap:
            for a in range(k):
                for b in range(a+1,k): pairs.append((ch[a],ch[b]))
        else:
            seen=set()
            while len(seen)<cap:
                a,b=sorted(map(int,rng.choice(k,size=2,replace=False)))
                seen.add((a,b))
            pairs.extend((ch[a],ch[b]) for a,b in seen)
    if len(pairs)>max_total:
        idx=rng.choice(len(pairs),size=max_total,replace=False)
        pairs=[pairs[int(i)] for i in idx]
    return pairs

def matched_controls(actual,centroids,masses,parent,ksearch,rmin,rmax,rng):
    from scipy.spatial import cKDTree
    if len(centroids)<3: return []
    tree=cKDTree(centroids)
    out=[]; used=set(); kk=min(len(centroids),max(3,ksearch))
    for i,j in actual:
        anchor=i if rng.random()<0.5 else j
        target=masses[j if anchor==i else i]
        _,idx=tree.query(centroids[anchor],k=kk)
        idx=np.atleast_1d(idx)
        chosen=None
        for c in idx:
            c=int(c)
            if c==anchor or parent[c]==parent[anchor]: continue
            ratio=masses[c]/max(1,target)
            key=tuple(sorted((anchor,c)))
            if rmin<=ratio<=rmax and key not in used:
                chosen=c;used.add(key);break
        if chosen is None:
            for c in idx:
                c=int(c); key=tuple(sorted((anchor,c)))
                if c!=anchor and parent[c]!=parent[anchor] and key not in used:
                    chosen=c;used.add(key);break
        if chosen is not None: out.append((anchor,chosen))
    return out

def accumulate_abs(means,pairs):
    ng=means.shape[1]
    s=np.zeros(ng);ss=np.zeros(ng)
    for a in range(0,len(pairs),512):
        q=pairs[a:a+512]
        if not q: continue
        A=means[[i for i,_ in q]].toarray()
        B=means[[j for _,j in q]].toarray()
        D=np.abs(A-B)
        s+=D.sum(axis=0);ss+=(D*D).sum(axis=0)
    return s,ss,len(pairs)

def detected_fraction(X):
    return np.asarray((X>0).mean(axis=0)).ravel()

def load_landmarks(project:Path,sample:str,cfg:dict):
    p=project/"results"/cfg["required_results"]["final_landscape"]/sample/"final_supported_pareto_landmarks.csv"
    d=pd.read_csv(p).sort_values("center_eval").reset_index(drop=True)
    base=project/"results"/cfg["required_results"]["merger_atlas"]/sample
    out=[]
    for i,r in d.iterrows():
        ee,en=int(r["entry_eval"]),int(r["entry_nodes"])
        ce,cn=int(r["center_eval"]),int(r["center_nodes"])
        ep=base/f"labels_entry_eval_{ee:03d}_nodes_{en}.npz"
        cp=base/f"labels_center_eval_{ce:03d}_nodes_{cn}.npz"
        if not ep.exists() or not cp.exists(): raise FileNotFoundError(f"Missing exact labels: {ep} / {cp}")
        out.append((i+1,r,np.load(ep)["labels"],np.load(cp)["labels"],ep,cp))
    return out

def score_merger_genes(project:Path,sample:str,cfg:dict,outdir:Path):
    g=cfg["gene_analysis"]; rng=np.random.default_rng(int(g["random_seed"]))
    X,xsrc=load_native_expression_aligned(project,sample)
    if X is None: raise RuntimeError(f"{sample}: exact expression alignment failed")
    genes=extract_gene_names(project/"data"/sample/"cell_feature_matrix.h5")
    if X.shape[1]!=len(genes): raise RuntimeError(f"{sample}: feature count mismatch")
    Xn=normalize_expression(X,g["normalization"])
    det=detected_fraction(X)
    keep=det>=float(g["min_detected_fraction"])
    P,psrc=load_coordinates(project,sample,X.shape[0])
    if P is None: raise RuntimeError(f"{sample}: exact coordinate alignment failed")
    sdir=ensure_dir(outdir/sample/"merger_biology")
    agg=[]
    nland=0
    for lid,r,entry,center,ep,cp in load_landmarks(project,sample,cfg):
        M,masses,codes,uniq=component_means(Xn,entry)
        cent=component_centroids(P,codes,len(uniq))
        par=parent_ids(codes,center)
        actual=sample_actual_pairs(par,int(g["max_pairs_per_parent"]),int(g["max_actual_pairs_per_landmark"]),rng)
        ctrl=matched_controls(actual,cent,masses,par,int(g["control_neighbors_to_search"]),
                              float(g["control_mass_ratio_min"]),float(g["control_mass_ratio_max"]),rng)
        sa,ssa,na=accumulate_abs(M,actual); sc,ssc,nc=accumulate_abs(M,ctrl)
        ma=sa/max(1,na); mc=sc/max(1,nc)
        va=np.maximum(0,ssa/max(1,na)-ma*ma); vc=np.maximum(0,ssc/max(1,nc)-mc*mc)
        se=np.sqrt(va/max(1,na)+vc/max(1,nc))
        score=mc-ma
        z=np.divide(score,se,out=np.zeros_like(score),where=se>0)
        tab=pd.DataFrame({
            "gene":genes,"detected_fraction":det,
            "merged_mean_abs_difference":ma,"control_mean_abs_difference":mc,
            "merger_concordance_score":score,"merger_concordance_z":z,
            "n_actual_pairs":na,"n_control_pairs":nc,
            "landmark_id":lid,"entry_nodes":int(r["entry_nodes"]),"center_nodes":int(r["center_nodes"]),
            "gene_passes_detection_filter":keep
        }).sort_values(["gene_passes_detection_filter","merger_concordance_z"],ascending=[False,False])
        tab.to_csv(sdir/f"landmark_{lid:02d}_gene_merger_scores.csv",index=False)
        q=tab[tab["gene_passes_detection_filter"]]
        for _,rr in q.iterrows():
            agg.append({"landmark_id":lid,"gene":rr["gene"],"score":rr["merger_concordance_score"],
                        "z":rr["merger_concordance_z"],"n_actual_pairs":na})
        nland+=1
    A=pd.DataFrame(agg)
    rows=[]
    if len(A):
        for gene,q in A.groupby("gene",sort=False):
            w=np.sqrt(np.maximum(1,q["n_actual_pairs"].to_numpy()))
            rows.append({
                "gene":gene,
                "weighted_mean_concordance_score":float(np.average(q["score"],weights=w)),
                "median_concordance_z":float(np.median(q["z"])),
                "positive_landmark_fraction":float(np.mean(q["score"]>0)),
                "n_landmarks":int(len(q))
            })
    summary=pd.DataFrame(rows)
    if len(summary):
        summary=summary.sort_values(["positive_landmark_fraction","median_concordance_z","weighted_mean_concordance_score"],
                                    ascending=[False,False,False])
    summary.to_csv(sdir/"gene_merger_score_summary.csv",index=False)
    meta={"sample":sample,"landmarks_scored":nland,"expression_source":xsrc,"coordinate_source":psrc,
          "hierarchy_changed":False,"new_merges_performed":False}
    (sdir/"merger_biology_certificate.json").write_text(json.dumps(meta,indent=2))
    return meta

def read_scientific_evaluations(project:Path,sample:str,cfg:dict):
    base=project/"results"/cfg["required_results"]["production_trajectory"]/sample
    p=base/"scientific_evaluations.parquet"
    if not p.exists():
        hits=sorted(base.rglob("scientific_evaluations*.parquet"))
        if not hits:return None,None
        p=hits[0]
    return pd.read_parquet(p),p

def find_col(cols,candidates):
    low={str(c).lower():c for c in cols}
    for x in candidates:
        if x in low:return low[x]
    return None
