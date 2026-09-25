from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd
from scipy import ndimage

@dataclass(frozen=True)
class FastGeometryConfig:
    pixel_size: float = 1.0
    min_interface_pixels: int = 4
    curvature_cap: float = 0.5
    curvature_max_points: int = 256

def file_sha256(path: Path, block=1<<20):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(block),b""):
            h.update(b)
    return h.hexdigest()

def config_hash(cfg):
    s=json.dumps(asdict(cfg),sort_keys=True).encode()
    return hashlib.sha256(s).hexdigest()

def background_components(mask):
    bg=(mask==0)
    lab,n=ndimage.label(bg,structure=np.ones((3,3),np.uint8))
    if n==0:
        return lab.astype(np.int32), pd.DataFrame(columns=[
            "background_component","kind","pixels","centroid_y","centroid_x"
        ])
    idx=np.arange(1,n+1)
    sizes=ndimage.sum(bg,lab,idx)
    cents=ndimage.center_of_mass(bg,lab,idx)
    border=np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]]))
    border=set(int(x) for x in border if x>0)
    rows=[]
    for k,size,cen in zip(idx,sizes,cents):
        rows.append({
            "background_component":int(k),
            "kind":"exterior" if int(k) in border else "internal_gap",
            "pixels":int(size),
            "centroid_y":float(cen[0]),
            "centroid_x":float(cen[1]),
        })
    return lab.astype(np.int32),pd.DataFrame(rows)

def _circle_curvature(xy,cap):
    if len(xy)<8: return 0.0,0.0
    if len(xy)>256:
        take=np.linspace(0,len(xy)-1,256).astype(int)
        xy=xy[take]
    x=xy[:,0].astype(float); y=xy[:,1].astype(float)
    x0=x.mean(); y0=y.mean(); u=x-x0; v=y-y0
    A=np.c_[2*u,2*v,np.ones_like(u)]; b=u*u+v*v
    try:
        sol,*_=np.linalg.lstsq(A,b,rcond=None)
        cx,cy,c=sol; r2=cx*cx+cy*cy+c
        if r2<=1e-12:return 0.0,0.0
        r=math.sqrt(r2); k=min(1.0/r,cap)
        radial=np.sqrt((u-cx)**2+(v-cy)**2)
        cv=float(np.std(radial)/max(np.mean(radial),1e-12))
        return float(k),float(np.exp(-5*cv))
    except Exception:
        return 0.0,0.0

def _edge_records(mask,bg_lab):
    """Vectorized extraction of every horizontal/vertical label transition."""
    # horizontal transitions
    a=mask[:,:-1]; b=mask[:,1:]
    q=(a!=b)&~((a==0)&(b==0))
    yh,xh=np.nonzero(q)
    ah=a[q].astype(np.int32); bh=b[q].astype(np.int32)
    # midpoint
    pxh=xh.astype(float)+1.0
    pyh=yh.astype(float)+0.5

    # vertical transitions
    a2=mask[:-1,:]; b2=mask[1:,:]
    q2=(a2!=b2)&~((a2==0)&(b2==0))
    yv,xv=np.nonzero(q2)
    av=a2[q2].astype(np.int32); bv=b2[q2].astype(np.int32)
    pxv=xv.astype(float)+0.5
    pyv=yv.astype(float)+1.0

    A=np.concatenate([ah,av]); B=np.concatenate([bh,bv])
    X=np.concatenate([pxh,pxv]); Y=np.concatenate([pyh,pyv])
    OR=np.concatenate([np.zeros(len(ah),np.int8),np.ones(len(av),np.int8)])

    kind=(A>0)&(B>0)
    ci=np.where(kind,np.minimum(A,B),np.where(A>0,A,B)).astype(np.int32)
    cj=np.where(kind,np.maximum(A,B),0).astype(np.int32)

    bc=np.zeros(len(A),np.int32)
    if len(ah):
        z=(ah==0)
        if z.any(): bc[:len(ah)][z]=bg_lab[yh[z],xh[z]]
        z=(bh==0)
        if z.any(): bc[:len(ah)][z]=bg_lab[yh[z],xh[z]+1]
    off=len(ah)
    if len(av):
        z=(av==0)
        if z.any(): bc[off:][z]=bg_lab[yv[z],xv[z]]
        z=(bv==0)
        if z.any(): bc[off:][z]=bg_lab[yv[z]+1,xv[z]]

    # int key columns: kind 1=cellcell 0=boundary, ci,cj,bc
    K=np.c_[kind.astype(np.int8),ci,cj,bc]
    return K,X,Y,OR

def extract_interfaces_fast(mask,cfg):
    bg_lab,bg_df=background_components(mask)
    K,X,Y,OR=_edge_records(mask,bg_lab)
    if len(K)==0:
        return pd.DataFrame(),bg_df,bg_lab

    order=np.lexsort((K[:,3],K[:,2],K[:,1],K[:,0]))
    K=K[order]; X=X[order]; Y=Y[order]; OR=OR[order]
    change=np.ones(len(K),bool)
    change[1:]=np.any(K[1:]!=K[:-1],axis=1)
    starts=np.flatnonzero(change)
    ends=np.r_[starts[1:],len(K)]

    rows=[]
    for s,e in zip(starts,ends):
        n=e-s
        if n<cfg.min_interface_pixels: continue
        kk=K[s]
        x=X[s:e]; y=Y[s:e]
        # second-moment tangent, no Python point loop
        mx=float(x.mean()); my=float(y.mean())
        ux=x-mx; uy=y-my
        cxx=float(np.dot(ux,ux)); cyy=float(np.dot(uy,uy)); cxy=float(np.dot(ux,uy))
        M=np.array([[cxx,cxy],[cxy,cyy]])
        vals,vecs=np.linalg.eigh(M); t=vecs[:,np.argmax(vals)]
        pts=np.c_[x,y]
        kappa,kconf=_circle_curvature(pts,cfg.curvature_cap)
        rows.append({
            "interface_id":len(rows),
            "kind":"cell_cell" if kk[0]==1 else "cell_boundary",
            "cell_i":int(kk[1]),
            "cell_j":int(kk[2]) if kk[0]==1 else None,
            "background_component":int(kk[3]) if kk[0]==0 else None,
            "length":float(n*cfg.pixel_size),
            "centroid_x":mx,"centroid_y":my,
            "tangent_x":float(t[0]),"tangent_y":float(t[1]),
            "curvature":float(kappa/max(cfg.pixel_size,1e-12)),
            "curvature_confidence":float(kconf),
            "n_interface_pixels":int(n),
        })
    return pd.DataFrame(rows),bg_df,bg_lab

def extract_junctions_fast(mask,interfaces,bg_lab):
    """
    Vectorized candidate detection on 2x2 blocks; only candidate junction blocks
    enter Python for interface-ID resolution.
    """
    # encode background components as negative ids; cells stay positive
    enc=np.where(mask>0,mask,-bg_lab).astype(np.int32)
    R=np.stack([enc[:-1,:-1],enc[:-1,1:],enc[1:,:-1],enc[1:,1:]],axis=-1)
    S=np.sort(R,axis=-1)
    uniq=1+np.sum(S[...,1:]!=S[...,:-1],axis=-1)
    ncell=np.sum(R>0,axis=-1)
    cand=(uniq>=3)&(ncell>=2)
    ys,xs=np.nonzero(cand)

    pair_to_e={}
    for r in interfaces.itertuples():
        if r.kind=="cell_cell":
            pair_to_e[("c",min(int(r.cell_i),int(r.cell_j)),max(int(r.cell_i),int(r.cell_j)))]=int(r.interface_id)
        else:
            pair_to_e[("b",int(r.cell_i),int(r.background_component))]=int(r.interface_id)

    rows=[]; seen=set()
    for y,x in zip(ys,xs):
        vals=sorted(set(int(v) for v in R[y,x]))
        inc=[]
        for i in range(len(vals)):
            for j in range(i+1,len(vals)):
                a,b=vals[i],vals[j]
                eid=None
                if a>0 and b>0:
                    eid=pair_to_e.get(("c",min(a,b),max(a,b)))
                elif (a>0) != (b>0):
                    c=a if a>0 else b
                    bc=abs(a if a<0 else b)
                    eid=pair_to_e.get(("b",c,bc))
                if eid is not None: inc.append(int(eid))
        inc=sorted(set(inc))
        if len(inc)<3: continue
        key=tuple(inc)
        if key in seen: continue
        seen.add(key)
        rows.append({
            "junction_id":len(rows),
            "x":float(x+1.0),"y":float(y+1.0),
            "incident_interfaces":inc,
            "n_regions":len(vals),
        })
    return pd.DataFrame(rows)

def extract_cell_centroids_fast(mask):
    maxlab=int(mask.max())
    ids=np.arange(1,maxlab+1)
    masses=ndimage.sum(np.ones(mask.shape,np.uint8),mask,ids)
    cents=ndimage.center_of_mass(np.ones(mask.shape,np.uint8),mask,ids)
    rows=[]
    for lab,m,c in zip(ids,masses,cents):
        if m<=0: continue
        rows.append({
            "cell_label":int(lab),
            "centroid_x":float(c[1]),"centroid_y":float(c[0]),
            "pixels":int(m)
        })
    return pd.DataFrame(rows)

def build_or_load_cache(mask_path:Path,cache_dir:Path,cfg:FastGeometryConfig):
    cache_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=cache_dir/"cache_manifest.json"
    mask_hash=file_sha256(mask_path)
    ch=config_hash(cfg)
    required=[
        cache_dir/"interfaces.parquet",
        cache_dir/"background_components.parquet",
        cache_dir/"background_labels.npz",
        cache_dir/"junctions.jsonl",
        cache_dir/"cells.parquet",
    ]
    if manifest_path.exists() and all(p.exists() for p in required):
        m=json.loads(manifest_path.read_text())
        if m.get("mask_sha256")==mask_hash and m.get("config_sha256")==ch:
            E=pd.read_parquet(required[0])
            B=pd.read_parquet(required[1])
            bg=np.load(required[2])["background_labels"]
            J=pd.read_json(required[3],lines=True)
            C=pd.read_parquet(required[4])
            return E,B,bg,J,C,True

    import tifffile
    mask=tifffile.imread(mask_path).astype(np.int32)
    E,B,bg=extract_interfaces_fast(mask,cfg)
    J=extract_junctions_fast(mask,E,bg)
    C=extract_cell_centroids_fast(mask)
    E.to_parquet(required[0],index=False)
    B.to_parquet(required[1],index=False)
    np.savez_compressed(required[2],background_labels=bg)
    J.to_json(required[3],orient="records",lines=True)
    C.to_parquet(required[4],index=False)
    manifest_path.write_text(json.dumps({
        "mask_sha256":mask_hash,
        "config_sha256":ch,
        "shape":list(mask.shape),
        "n_interfaces":len(E),
        "n_junctions":len(J),
        "n_cells":len(C),
    },indent=2))
    return E,B,bg,J,C,False
