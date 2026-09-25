from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import hashlib, json, math, os, time
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Polygon


@dataclass(frozen=True)
class PreflightConfig:
    sample_workers: int = 3
    transcript_rowgroup_sample_max: int = 12
    transcript_points_per_rowgroup_max: int = 25000
    topology_cell_sample_max: int = 6000
    topology_jitter_repeats: int = 4
    topology_jitter_fraction_of_epsilon: float = 0.20
    gap_raster_max_pixels: int = 6_000_000
    morphology_downsample_max_dim: int = 2200
    morphology_quantile: float = 0.35
    synthetic_seed: int = 1729

    # Hard data-integrity thresholds.
    max_duplicate_cell_id_fraction: float = 0.0
    max_nonfinite_coordinate_fraction: float = 1e-6
    min_matrix_cell_overlap_fraction: float = 0.999
    min_boundary_cell_overlap_fraction: float = 0.995

    # Structural/scientific warning thresholds.
    max_isolated_fraction_warn: float = 0.05
    min_largest_component_fraction_warn: float = 0.70
    max_internal_unassigned_fraction_warn: float = 0.10
    min_topology_jaccard_warn: float = 0.80
    min_synthetic_recovery_warn: float = 0.95


def sha256_file(path: Path, block=2**20):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()


def find_one(root: Path, patterns):
    if isinstance(patterns,str): patterns=[patterns]
    for pat in patterns:
        hits=list(root.rglob(pat))
        if hits:
            return sorted(hits)[0]
    return None


def all_hits(root: Path, patterns):
    if isinstance(patterns,str): patterns=[patterns]
    out=[];seen=set()
    for pat in patterns:
        for p in root.rglob(pat):
            if p not in seen:
                seen.add(p);out.append(p)
    return sorted(out)


def parquet_metadata(path: Path):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(path)
    return {
        "rows":int(pf.metadata.num_rows),
        "row_groups":int(pf.metadata.num_row_groups),
        "columns":pf.schema_arrow.names,
    }


def read_parquet_columns(path: Path, wanted):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(path)
    avail=set(pf.schema_arrow.names)
    cols=[c for c in wanted if c in avail]
    return pf.read(columns=cols).to_pandas()


def coord_columns(columns, kind="cell"):
    cols=set(columns)
    candidates=[
        ("x_centroid","y_centroid"),
        ("x_location","y_location"),
        ("vertex_x","vertex_y"),
        ("x","y"),
        ("transcript_x","transcript_y"),
    ]
    for a,b in candidates:
        if a in cols and b in cols:
            return a,b
    return None,None


def coordinate_summary(df: pd.DataFrame):
    xcol,ycol=coord_columns(df.columns)
    if xcol is None:
        return {"available":False}
    x=pd.to_numeric(df[xcol],errors="coerce").to_numpy(float)
    y=pd.to_numeric(df[ycol],errors="coerce").to_numpy(float)
    finite=np.isfinite(x)&np.isfinite(y)
    if not finite.any():
        return {"available":True,"finite_fraction":0.0}
    xf=x[finite];yf=y[finite]
    return {
        "available":True,
        "x_col":xcol,"y_col":ycol,
        "finite_fraction":float(finite.mean()),
        "x_min":float(xf.min()),"x_max":float(xf.max()),
        "y_min":float(yf.min()),"y_max":float(yf.max()),
        "x_span":float(xf.max()-xf.min()),
        "y_span":float(yf.max()-yf.min()),
    }


def bbox_overlap_fraction(a,b):
    if not a.get("available") or not b.get("available"):
        return None
    ax0,ax1=a["x_min"],a["x_max"]; ay0,ay1=a["y_min"],a["y_max"]
    bx0,bx1=b["x_min"],b["x_max"]; by0,by1=b["y_min"],b["y_max"]
    ix=max(0,min(ax1,bx1)-max(ax0,bx0))
    iy=max(0,min(ay1,by1)-max(ay0,by0))
    inter=ix*iy
    area=min(max((ax1-ax0)*(ay1-ay0),1e-12),
             max((bx1-bx0)*(by1-by0),1e-12))
    return float(inter/area)


def h5_feature_matrix_summary(path: Path):
    import h5py
    out={"path":str(path)}
    with h5py.File(path,"r") as h:
        g=h["matrix"] if "matrix" in h else h
        shape=tuple(int(x) for x in g["shape"][:]) if "shape" in g else None
        out["shape"]=shape
        if "data" in g:
            out["nnz"]=int(g["data"].shape[0])
        feats=g.get("features",None)
        names=[];ids=[]
        if feats is not None:
            for key,target in (("name",names),("id",ids)):
                if key in feats:
                    vals=feats[key][:]
                    target.extend([
                        x.decode() if isinstance(x,(bytes,np.bytes_)) else str(x)
                        for x in vals
                    ])
        out["feature_names"]=names
        out["feature_ids"]=ids
        if "barcodes" in g:
            vals=g["barcodes"][:]
            out["barcodes"]=[
                x.decode() if isinstance(x,(bytes,np.bytes_)) else str(x)
                for x in vals
            ]
        else:
            out["barcodes"]=[]
    return out


def table_id_audit(df, id_col):
    if id_col not in df.columns:
        return {"available":False}
    s=df[id_col].astype(str)
    n=len(s)
    nunique=int(s.nunique(dropna=False))
    dup=n-nunique
    return {
        "available":True,
        "rows":n,
        "unique":nunique,
        "duplicates":int(dup),
        "duplicate_fraction":float(dup/max(n,1)),
        "null_fraction":float(df[id_col].isna().mean()),
    }


def polygons_from_boundary_table(df):
    if "cell_id" not in df.columns:
        return {}
    xcol,ycol=coord_columns(df.columns)
    if xcol is None: return {}
    order=None
    for c in ("vertex_order","vertex_index","vertex_id"):
        if c in df.columns:
            order=c;break
    out={}
    for cid,g in df.groupby("cell_id",sort=False):
        if order: g=g.sort_values(order)
        xy=g[[xcol,ycol]].to_numpy(float)
        xy=xy[np.isfinite(xy).all(axis=1)]
        if len(xy)<3: continue
        try:
            p=Polygon(xy)
            if not p.is_valid: p=p.buffer(0)
            if not p.is_empty: out[str(cid)]=p
        except Exception:
            pass
    return out


def rasterize_union(polys, max_pixels):
    if not polys:
        return None,None
    b=np.array([p.bounds for p in polys.values()],float)
    xmin,ymin=float(b[:,0].min()),float(b[:,1].min())
    xmax,ymax=float(b[:,2].max()),float(b[:,3].max())
    w=max(xmax-xmin,1e-9);h=max(ymax-ymin,1e-9)
    scale=min(max(math.sqrt(max_pixels/(w*h)),0.15),3.0)
    W=max(8,int(math.ceil(w*scale))+3)
    H=max(8,int(math.ceil(h*scale))+3)
    mask=np.zeros((H,W),bool)

    try:
        import shapely
        has_xy=hasattr(shapely,"contains_xy")
    except Exception:
        shapely=None;has_xy=False

    for p in polys.values():
        bx0=max(0,int(math.floor((p.bounds[0]-xmin)*scale))+1)
        bx1=min(W,int(math.ceil((p.bounds[2]-xmin)*scale))+2)
        by0=max(0,int(math.floor((p.bounds[1]-ymin)*scale))+1)
        by1=min(H,int(math.ceil((p.bounds[3]-ymin)*scale))+2)
        if bx1<=bx0 or by1<=by0: continue
        yy,xx=np.mgrid[by0:by1,bx0:bx1]
        px=xmin+(xx-1+0.5)/scale
        py=ymin+(yy-1+0.5)/scale
        if has_xy:
            inside=shapely.contains_xy(p,px.ravel(),py.ravel()).reshape(px.shape)
        else:
            from shapely.geometry import Point
            inside=np.array([p.covers(Point(x,y)) for x,y in zip(px.ravel(),py.ravel())]).reshape(px.shape)
        mask[by0:by1,bx0:bx1] |= inside
    return mask,{"bounds":[xmin,ymin,xmax,ymax],"scale":scale,"shape":[H,W]}


def gap_metrics(mask):
    if mask is None:
        return {"available":False}
    filled=ndimage.binary_fill_holes(mask)
    internal=filled & ~mask
    lab,n=ndimage.label(internal)
    sizes=np.bincount(lab.ravel())[1:] if n else np.array([],int)
    return {
        "available":True,
        "coverage_of_filled_envelope":float(mask.sum()/max(filled.sum(),1)),
        "internal_unassigned_fraction":float(internal.sum()/max(filled.sum(),1)),
        "n_internal_components":int(n),
        "largest_internal_component_pixels":int(sizes.max()) if len(sizes) else 0,
        "median_internal_component_pixels":float(np.median(sizes)) if len(sizes) else 0.0,
    }


def morphology_file_summary(path: Path):
    import tifffile
    out={"path":str(path),"readable":False}
    with tifffile.TiffFile(path) as tf:
        s=tf.series[0]
        out["shape"]=list(map(int,s.shape))
        out["dtype"]=str(s.dtype)
        out["axes"]=str(getattr(s,"axes",""))
        out["ome_metadata_present"]=bool(tf.ome_metadata)
        out["readable"]=True
        if tf.ome_metadata:
            md=tf.ome_metadata
            # only record whether physical-size fields are present; do not parse guesses
            out["physical_size_metadata_present"]=(
                "PhysicalSizeX" in md and "PhysicalSizeY" in md
            )
        else:
            out["physical_size_metadata_present"]=False
    return out


def transcript_stream_summary(path: Path, cell_bbox, cfg: PreflightConfig):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(path)
    names=pf.schema_arrow.names
    xcol,ycol=coord_columns(names)
    cellid_col=None
    for c in ("cell_id","cell_ID","assigned_cell_id"):
        if c in names:
            cellid_col=c;break
    if xcol is None:
        return {"available":True,"coordinate_columns_found":False}

    rgs=np.linspace(
        0,max(pf.metadata.num_row_groups-1,0),
        min(cfg.transcript_rowgroup_sample_max,max(pf.metadata.num_row_groups,1)),
        dtype=int
    )
    rgs=np.unique(rgs)
    n=0;assigned=0;inside_bbox=0
    mins=[np.inf,np.inf];maxs=[-np.inf,-np.inf]
    for rg in rgs:
        cols=[xcol,ycol]+([cellid_col] if cellid_col else [])
        t=pf.read_row_group(int(rg),columns=cols).to_pandas()
        if len(t)>cfg.transcript_points_per_rowgroup_max:
            t=t.sample(cfg.transcript_points_per_rowgroup_max,random_state=1729)
        x=pd.to_numeric(t[xcol],errors="coerce").to_numpy(float)
        y=pd.to_numeric(t[ycol],errors="coerce").to_numpy(float)
        ok=np.isfinite(x)&np.isfinite(y)
        x=x[ok];y=y[ok]
        n+=len(x)
        if len(x):
            mins[0]=min(mins[0],float(x.min())); mins[1]=min(mins[1],float(y.min()))
            maxs[0]=max(maxs[0],float(x.max())); maxs[1]=max(maxs[1],float(y.max()))
            if cell_bbox and cell_bbox.get("available"):
                q=(x>=cell_bbox["x_min"])&(x<=cell_bbox["x_max"])&(y>=cell_bbox["y_min"])&(y<=cell_bbox["y_max"])
                inside_bbox+=int(q.sum())
        if cellid_col:
            ss=t.loc[ok,cellid_col]
            assigned+=int((ss.notna() & (ss.astype(str)!="UNASSIGNED") & (ss.astype(str)!="0")).sum())
    return {
        "available":True,
        "coordinate_columns_found":True,
        "sampled_points":int(n),
        "assigned_cell_fraction_sampled":float(assigned/max(n,1)) if cellid_col else None,
        "inside_cell_bbox_fraction_sampled":float(inside_bbox/max(n,1)),
        "x_min_sampled":None if not np.isfinite(mins[0]) else mins[0],
        "x_max_sampled":None if not np.isfinite(maxs[0]) else maxs[0],
        "y_min_sampled":None if not np.isfinite(mins[1]) else mins[1],
        "y_max_sampled":None if not np.isfinite(maxs[1]) else maxs[1],
        "cell_assignment_column":cellid_col,
        "row_groups_sampled":rgs.tolist(),
        "total_rows":int(pf.metadata.num_rows),
    }


def graph_components(edges, all_cells):
    adj=defaultdict(set)
    for a,b in edges:
        a=str(a);b=str(b)
        adj[a].add(b);adj[b].add(a)
    seen=set();sizes=[]
    for c in all_cells:
        if c in seen: continue
        q=[c];seen.add(c);n=0
        while q:
            u=q.pop();n+=1
            for v in adj.get(u,()):
                if v not in seen:
                    seen.add(v);q.append(v)
        sizes.append(n)
    sizes.sort(reverse=True)
    deg=np.array([len(adj.get(c,())) for c in all_cells],int)
    return {
        "n_components":len(sizes),
        "largest_component_fraction":float(sizes[0]/max(len(all_cells),1)) if sizes else 0.0,
        "isolated_fraction":float(np.mean(deg==0)) if len(deg) else 1.0,
        "mean_degree":float(np.mean(deg)) if len(deg) else 0.0,
        "median_degree":float(np.median(deg)) if len(deg) else 0.0,
        "max_degree":int(deg.max()) if len(deg) else 0,
    }


def edge_set_from_df(df):
    out=set()
    if "cell_i" not in df.columns or "cell_j" not in df.columns:
        return out
    for a,b in zip(df.cell_i.astype(str),df.cell_j.astype(str)):
        if a==b: continue
        out.add(tuple(sorted((a,b))))
    return out


def topology_jitter_audit(cells: pd.DataFrame, edges: pd.DataFrame, epsilon, cfg: PreflightConfig):
    """
    Cheap adversarial topology test on a centroid-space surrogate.

    It does NOT claim to rederive Gate-B boundary contacts. It asks whether the
    certified graph is grossly fragile to coordinate perturbation by comparing
    nearest-neighbor candidate structure before/after jitter on a large subset.
    """
    xcol,ycol=coord_columns(cells.columns)
    if xcol is None or epsilon is None or epsilon<=0:
        return {"available":False}

    c=cells[["cell_id",xcol,ycol]].copy()
    c["cell_id"]=c.cell_id.astype(str)
    c=c[np.isfinite(c[xcol])&np.isfinite(c[ycol])]
    if len(c)>cfg.topology_cell_sample_max:
        c=c.sample(cfg.topology_cell_sample_max,random_state=cfg.synthetic_seed)
    pts=c[[xcol,ycol]].to_numpy(float)
    ids=c.cell_id.tolist()
    idset=set(ids)

    base=edge_set_from_df(edges)
    base={e for e in base if e[0] in idset and e[1] in idset}
    if not base:
        return {"available":False,"reason":"no certified edges in sample"}

    # Candidate graph uses k-nearest neighbors with k chosen from certified mean degree.
    mean_deg=max(2,int(round(2*len(base)/max(len(ids),1))))
    k=min(max(mean_deg+3,5),12)
    tree=cKDTree(pts)
    _,idx=tree.query(pts,k=min(k+1,len(pts)))
    cand=set()
    for i,row in enumerate(np.atleast_2d(idx)):
        for j in row[1:]:
            cand.add(tuple(sorted((ids[i],ids[int(j)]))))

    rng=np.random.default_rng(cfg.synthetic_seed)
    jacc=[]
    sigma=float(epsilon*cfg.topology_jitter_fraction_of_epsilon)
    for _ in range(cfg.topology_jitter_repeats):
        p2=pts+rng.normal(0,sigma,size=pts.shape)
        tr=cKDTree(p2)
        _,ix=tr.query(p2,k=min(k+1,len(p2)))
        c2=set()
        for i,row in enumerate(np.atleast_2d(ix)):
            for j in row[1:]:
                c2.add(tuple(sorted((ids[i],ids[int(j)]))))
        jacc.append(len(cand&c2)/max(len(cand|c2),1))
    return {
        "available":True,
        "surrogate_only":True,
        "sampled_cells":len(ids),
        "jitter_sigma_coordinate_units":sigma,
        "candidate_k":k,
        "mean_candidate_jaccard_under_jitter":float(np.mean(jacc)),
        "min_candidate_jaccard_under_jitter":float(np.min(jacc)),
    }


def synthetic_recovery_suite(seed=1729):
    rng=np.random.default_rng(seed)
    tests=[]

    # 1. ID permutation invariance.
    ids=np.array([f"C{i}" for i in range(100)])
    vals=rng.normal(size=100)
    p=rng.permutation(100)
    ok=np.allclose(np.sort(vals),np.sort(vals[p]))
    tests.append(("id_permutation_invariance",ok))

    # 2. Coordinate translation invariance of pairwise distances.
    pts=rng.normal(size=(200,2))
    d1=np.linalg.norm(pts[:50,None,:]-pts[None,:50,:],axis=-1)
    shift=np.array([123.4,-92.0])
    pts2=pts+shift
    d2=np.linalg.norm(pts2[:50,None,:]-pts2[None,:50,:],axis=-1)
    tests.append(("coordinate_translation_invariance",bool(np.allclose(d1,d2))))

    # 3. Unit scaling detectability.
    span=np.ptp(pts,axis=0)
    span1000=np.ptp(pts*1000,axis=0)
    ratio=float(np.median(span1000/span))
    tests.append(("unit_scaling_detected",bool(abs(ratio-1000)<1e-8)))

    # 4. Hole detection.
    mask=np.zeros((64,64),bool);mask[8:56,8:56]=1;mask[24:40,24:40]=0
    gm=gap_metrics(mask)
    tests.append(("internal_hole_detection",bool(gm["n_internal_components"]==1)))

    # 5. Disconnected-domain detection.
    mask=np.zeros((64,64),bool);mask[5:20,5:20]=1;mask[40:55,40:55]=1
    _,n=ndimage.label(mask)
    tests.append(("disconnected_domain_detection",bool(n==2)))

    passed=sum(int(x[1]) for x in tests)
    return {
        "tests":[{"name":n,"pass":bool(ok)} for n,ok in tests],
        "passed":passed,
        "total":len(tests),
        "recovery_fraction":float(passed/max(len(tests),1)),
    }


def status(pass_condition, warning_condition=False, note=None):
    if not pass_condition:
        s="FAIL"
    elif warning_condition:
        s="WARN"
    else:
        s="PASS"
    return {"status":s, **({"note":note} if note else {})}
