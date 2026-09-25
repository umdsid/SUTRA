from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
from scipy import ndimage

ST4=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
ST8=np.ones((3,3),dtype=np.uint8)

def sha256(path:Path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def components_per_label(mask,connectivity=1):
    st=ST4 if connectivity==1 else ST8
    maxlab=int(mask.max())
    slices=ndimage.find_objects(mask,max_label=maxlab)
    out={}
    for lab in range(1,maxlab+1):
        sl=slices[lab-1] if lab-1<len(slices) else None
        if sl is None: continue
        _,n=ndimage.label(mask[sl]==lab,structure=st)
        out[lab]=int(n)
    return out

def disconnected(mask,connectivity=1):
    return [k for k,v in components_per_label(mask,connectivity).items() if v>1]

def boundary_relabel_like_readme(img):
    from skimage.segmentation import find_boundaries
    from skimage.measure import label
    b=find_boundaries(img,mode="subpixel")
    return label(1-b,connectivity=1)

def upsample_labels_to_subpixel_grid(img):
    """
    Map an N x M label image to the (2N-1) x (2M-1) grid used by
    find_boundaries(..., mode='subpixel').

    Original pixel centers occupy even/even indices. Intermediate positions are
    left zero because they do not correspond to original pixels.
    """
    img=np.asarray(img)
    H,W=img.shape
    out=np.zeros((2*H-1,2*W-1),dtype=img.dtype)
    out[0::2,0::2]=img
    return out

def faq_cleanup_like_readme(img,min_size=10,expand_distance=5):
    from skimage.measure import label,regionprops_table
    from skimage.segmentation import expand_labels
    import pandas as pd
    out=np.asarray(img).copy()
    labels=label(out,connectivity=1)
    props=pd.DataFrame(regionprops_table(labels,properties=("label","area")))
    exterior=(out==0)
    small=props.loc[props["area"]<min_size,"label"].to_numpy()
    if len(small):
        out[np.isin(labels,small)]=0
        out=expand_labels(out,distance=expand_distance)
        out[exterior]=0
        out=label(out,connectivity=1)
    return out

def contingency(a,b):
    a=np.asarray(a,dtype=np.int64).ravel()
    b=np.asarray(b,dtype=np.int64).ravel()
    if a.shape!=b.shape:
        raise ValueError(f"contingency shape mismatch: {a.shape} vs {b.shape}")
    q=(a>0)&(b>0)
    if not q.any(): return {}
    pairs=np.c_[a[q],b[q]]
    uniq,counts=np.unique(pairs,axis=0,return_counts=True)
    out={}
    for (x,y),n in zip(uniq,counts):
        out.setdefault(int(x),[]).append((int(y),int(n)))
    return out

def dominant_mapping(a,b):
    c=contingency(a,b)
    return {x:max(vals,key=lambda z:z[1])[0] for x,vals in c.items()}

def mapped_component_audit(original,transformed,original_bad,subpixel=False):
    ref=upsample_labels_to_subpixel_grid(original) if subpixel else original
    m=dominant_mapping(ref,transformed)
    tcomp=components_per_label(transformed,1)
    rows=[]
    for lab in original_bad:
        tlab=m.get(int(lab))
        rows.append({
            "original_label":int(lab),
            "mapped_label":None if tlab is None else int(tlab),
            "mapped_components_4":None if tlab is None else int(tcomp.get(tlab,0)),
        })
    return rows

def source_manifest(tensionmap_root:Path):
    files=[]
    for rel in ("src/VMSI.py","src/segment.py","README.md"):
        p=tensionmap_root/rel
        if p.exists():
            files.append({"path":rel,"sha256":sha256(p),"bytes":p.stat().st_size})
    return files
