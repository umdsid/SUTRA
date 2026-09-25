from __future__ import annotations
import re
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

X_NAMES=("centroid_x","x_centroid","center_x","x_center","x_location","x_coord","x")
Y_NAMES=("centroid_y","y_centroid","center_y","y_center","y_location","y_coord","y")
TENSION_NAMES=("tension_z","tau_z","tension","tau","interface_tension_z")
DP_NAMES=("delta_pressure_z","delta_p_z","pressure_difference_z")

def resolve_xy(cells):
    cols=set(cells.columns)
    x=next((c for c in X_NAMES if c in cols),None)
    y=next((c for c in Y_NAMES if c in cols),None)
    if x is None or y is None:
        raise RuntimeError(f"cannot resolve centroids from {list(cells.columns)}")
    return x,y

def canonical_edges(i,j):
    i=np.asarray(i,np.int64); j=np.asarray(j,np.int64)
    z=np.stack([np.minimum(i,j),np.maximum(i,j)],axis=1)
    z=z[z[:,0]!=z[:,1]]
    return np.unique(z,axis=0) if len(z) else np.empty((0,2),np.int64)

def build_backbone(xy,contacts,k=16,gap_factor=2.25,local_k=4):
    xy=np.asarray(xy,float); n=len(xy)
    tree=cKDTree(xy)
    kk=min(max(k+1,local_k+1),n)
    dist,ind=tree.query(xy,k=kk)
    if dist.ndim==1: dist=dist[:,None]; ind=ind[:,None]
    lk=min(local_k,dist.shape[1]-1)
    local=np.maximum(dist[:,lk],1e-12)
    knn=[]
    for i in range(n):
        for q in range(1,min(k,dist.shape[1]-1)+1):
            j=int(ind[i,q]); d=float(dist[i,q])
            if d <= gap_factor*max(local[i],local[j]):
                knn.append((min(i,j),max(i,j)))
    K=np.unique(np.asarray(knn,np.int64),axis=0) if knn else np.empty((0,2),np.int64)
    C=canonical_edges(contacts[:,0],contacts[:,1])
    E=np.unique(np.vstack([C,K]),axis=0)
    cs=set(map(tuple,C.tolist())); ks=set(map(tuple,K.tolist()))
    out=pd.DataFrame(E,columns=["cell_i_index","cell_j_index"])
    out["source_contact"]=[tuple(x) in cs for x in E]
    out["source_knn"]=[tuple(x) in ks for x in E]
    dxy=xy[E[:,0]]-xy[E[:,1]]
    out["spatial_distance"]=np.linalg.norm(dxy,axis=1)
    out["local_scale_i"]=local[E[:,0]]
    out["local_scale_j"]=local[E[:,1]]
    out["gap_guard_ratio"]=out.spatial_distance/np.maximum(np.maximum(out.local_scale_i,out.local_scale_j),1e-12)
    return out

def cosine_similarity_edges(Y,edges,chunk=50000):
    Y=np.asarray(Y,np.float32)
    norms=np.linalg.norm(Y,axis=1).astype(float)
    i=edges.cell_i_index.to_numpy(np.int64); j=edges.cell_j_index.to_numpy(np.int64)
    out=np.empty(len(edges),float)
    for a in range(0,len(edges),chunk):
        z=min(len(edges),a+chunk)
        A=Y[i[a:z]].astype(float,copy=False); B=Y[j[a:z]].astype(float,copy=False)
        num=np.einsum("ij,ij->i",A,B)
        den=norms[i[a:z]]*norms[j[a:z]]
        out[a:z]=np.divide(num,den,out=np.zeros(z-a),where=den>0)
    return np.clip(out,-1,1)

def spatial_expression_weight(edges,cosine):
    d=edges.spatial_distance.to_numpy(float)
    r=np.maximum(np.maximum(edges.local_scale_i.to_numpy(float),edges.local_scale_j.to_numpy(float)),1e-12)
    ds=d/r
    de=1.0-np.asarray(cosine,float)
    p=de[(de>0)&np.isfinite(de)]
    escale=max(float(np.median(p)) if len(p) else 1.0,1e-6)
    return np.exp(-(ds*ds)-(de/escale)**2),ds,de,escale

def functional_distance_edges(Y,edges,G,chunk=12000):
    Y=np.asarray(Y,np.float32); G=np.asarray(G,float)
    i=edges.cell_i_index.to_numpy(np.int64); j=edges.cell_j_index.to_numpy(np.int64)
    out=np.empty(len(edges),float)
    for a in range(0,len(edges),chunk):
        z=min(len(edges),a+chunk)
        D=Y[i[a:z]].astype(float)-Y[j[a:z]].astype(float)
        out[a:z]=np.sqrt(np.maximum(0,np.einsum("ij,jk,ik->i",D,G,D,optimize=True)))
    return out

def adjacency_from_edges(edges,n,weight=None):
    i=edges.cell_i_index.to_numpy(np.int64); j=edges.cell_j_index.to_numpy(np.int64)
    w=np.ones(len(edges),float) if weight is None else np.asarray(weight,float)
    A=sparse.coo_matrix((np.r_[w,w],(np.r_[i,j],np.r_[j,i])),shape=(n,n)).tocsr()
    A.sum_duplicates(); return A

def harmonic_fill(A,anchor_mask,anchor_values,prior=0.0,max_iter=500,tol=1e-7):
    A=sparse.csr_matrix(A,dtype=float)
    anchor=np.asarray(anchor_mask,bool); vals=np.asarray(anchor_values,float)
    if anchor.any() and not np.isfinite(prior):
        prior=float(np.nanmedian(vals[anchor]))
    x=np.full(A.shape[0],prior,float); x[anchor]=vals[anchor]
    deg=np.asarray(A.sum(axis=1)).ravel()
    P=sparse.diags(np.divide(1,deg,out=np.zeros_like(deg),where=deg>0))@A
    free=~anchor
    for _ in range(max_iter):
        y=P@x; y[anchor]=vals[anchor]; y[deg==0]=prior
        err=float(np.max(np.abs(y[free]-x[free]))) if free.any() else 0.0
        x=y
        if err<=tol: break
    # confidence by graph hops from anchors
    U=(A>0).astype(np.int8)
    hops=np.full(A.shape[0],np.inf); hops[anchor]=0
    seen=anchor.copy(); frontier=anchor.copy()
    for h in range(1,65):
        if not frontier.any(): break
        nbr=np.asarray((U@frontier.astype(np.int8))>0).ravel()
        new=nbr & ~seen
        hops[new]=h; seen|=new; frontier=new
    conf=np.exp(-np.minimum(hops,64)/8.0); conf[~np.isfinite(hops)]=0; conf[anchor]=1
    prov=np.full(A.shape[0],"propagated",object); prov[anchor]="observed_anchor"; prov[conf==0]="neutral_prior"
    return x,conf,prov

def flexible_col(df,names):
    return next((c for c in names if c in df.columns),None)

def edge_anchor_to_nodes(n,edges,value_col,valid_col=None):
    if value_col is None: return np.zeros(n,bool),np.full(n,np.nan)
    v=edges[value_col].to_numpy(float)
    q=np.isfinite(v)
    if valid_col and valid_col in edges.columns: q &= edges[valid_col].astype(bool).to_numpy()
    i=edges.cell_i_index.to_numpy(np.int64)[q]; j=edges.cell_j_index.to_numpy(np.int64)[q]; vv=v[q]
    s=np.zeros(n,float); c=np.zeros(n,float)
    np.add.at(s,i,vv); np.add.at(s,j,vv); np.add.at(c,i,1); np.add.at(c,j,1)
    out=np.divide(s,c,out=np.full(n,np.nan),where=c>0)
    return c>0,out

def parse_complex(s):
    if s is None or (isinstance(s,float) and np.isnan(s)): return ()
    parts=[x for x in re.split(r"[_+&|:/\s]+",str(s)) if x]
    return tuple(dict.fromkeys(parts))

def compile_cellchat(registry,genes):
    """Compile panel-supported directed CellChat channels.

    STRATA's canonical typed registry uses sender_gene / receiver_gene.
    Generic CellChat-derived tables may instead use ligand / receptor naming.
    Both are accepted, with direction preserved exactly as source -> target.
    """
    gmap={str(g).upper():i for i,g in enumerate(genes)}

    sender_candidates=[
        "sender_gene",
        "ligand",
        "ligand_gene",
        "ligand_genes",
        "source_gene",
    ]
    receiver_candidates=[
        "receiver_gene",
        "receptor",
        "receptor_gene",
        "receptor_genes",
        "target_gene",
    ]

    lc=next((c for c in sender_candidates if c in registry.columns),None)
    rc=next((c for c in receiver_candidates if c in registry.columns),None)

    if lc is None or rc is None:
        raise RuntimeError(
            "CellChat columns unsupported: "
            f"{list(registry.columns)}; "
            f"expected sender in {sender_candidates} and receiver in {receiver_candidates}"
        )

    out=[]
    for idx,r in registry.iterrows():
        lg=parse_complex(r[lc])
        rg=parse_complex(r[rc])

        li=[gmap[g.upper()] for g in lg if g.upper() in gmap]
        ri=[gmap[g.upper()] for g in rg if g.upper() in gmap]

        # Complexes are panel-supported only if every named component is present.
        if lg and rg and len(li)==len(lg) and len(ri)==len(rg):
            out.append((int(idx),tuple(li),tuple(ri)))

    return out

def channel_activity(Y,channels,side):
    Y=np.asarray(Y,np.float32); A=np.empty((Y.shape[0],len(channels)),np.float32)
    for k,ch in enumerate(channels):
        inds=ch[1] if side=="ligand" else ch[2]
        X=np.maximum(Y[:,inds],0)
        A[:,k]=X[:,0] if len(inds)==1 else np.exp(np.mean(np.log(X+1e-8),axis=1)).astype(np.float32)
    return A

def communication_edges(Y,edges,channels,chunk=12000):
    if not channels:
        z=np.zeros(len(edges),float); return z,z,z,z
    L=channel_activity(Y,channels,"ligand"); R=channel_activity(Y,channels,"receptor")
    i=edges.cell_i_index.to_numpy(np.int64); j=edges.cell_j_index.to_numpy(np.int64)
    f=np.empty(len(edges)); r=np.empty(len(edges)); sf=np.empty(len(edges)); sr=np.empty(len(edges))
    for a in range(0,len(edges),chunk):
        z=min(len(edges),a+chunk)
        P=L[i[a:z]]*R[j[a:z]]; Q=L[j[a:z]]*R[i[a:z]]
        f[a:z]=P.mean(axis=1); r[a:z]=Q.mean(axis=1)
        sf[a:z]=(P>0).mean(axis=1); sr[a:z]=(Q>0).mean(axis=1)
    return f,r,sf,sr
