from __future__ import annotations

from collections import defaultdict, deque
import math
import numpy as np
import pandas as pd


def _safe_cv(x):
    x=np.asarray(x,float)
    x=x[np.isfinite(x)]
    if len(x)==0: return np.nan
    m=float(np.mean(x))
    return float(np.std(x)/max(abs(m),1e-12))


def _entropy(vals, bins=12, period=None):
    x=np.asarray(vals,float)
    x=x[np.isfinite(x)]
    if len(x)<2: return 0.0
    if period is not None:
        x=np.mod(x,period)
        hist,_=np.histogram(x,bins=bins,range=(0,period))
    else:
        hist,_=np.histogram(x,bins=bins)
    p=hist.astype(float)
    p=p[p>0]
    p/=p.sum()
    h=-np.sum(p*np.log(p))
    return float(h/max(np.log(bins),1e-12))


def _graph_features(cells, E):
    cells=sorted(set(int(x) for x in cells))
    cset=set(cells)
    adj={c:set() for c in cells}
    n_edges=0
    for r in E.itertuples():
        if r.kind!="cell_cell": continue
        a,b=int(r.cell_i),int(r.cell_j)
        if a in cset and b in cset and a!=b:
            if b not in adj[a]:
                adj[a].add(b); adj[b].add(a); n_edges+=1
    deg=np.array([len(adj[c]) for c in cells],float)

    # connected components
    seen=set(); comps=[]
    for s in cells:
        if s in seen: continue
        q=[s]; seen.add(s); cc=[]
        while q:
            u=q.pop(); cc.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        comps.append(cc)

    # exact graph diameter on each component, feasible for ~300-cell patches
    diameter=0
    for cc in comps:
        for s in cc:
            dist={s:0}; q=deque([s])
            while q:
                u=q.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v]=dist[u]+1; q.append(v)
            if dist:
                diameter=max(diameter,max(dist.values()))

    # clustering
    local=[]
    triangles_times3=0
    for u in cells:
        nb=list(adj[u]); k=len(nb)
        if k<2:
            local.append(0.0); continue
        links=0
        for i in range(k):
            for j in range(i+1,k):
                if nb[j] in adj[nb[i]]:
                    links+=1
        local.append(2*links/(k*(k-1)))
        triangles_times3 += links

    n=len(cells)
    density=(2*n_edges/(n*(n-1))) if n>1 else 0.0
    cycle_rank=n_edges-n+len(comps)
    return {
        "graph_n_edges":int(n_edges),
        "graph_mean_degree":float(np.mean(deg)) if len(deg) else 0.0,
        "graph_degree_cv":_safe_cv(deg),
        "graph_degree_variance":float(np.var(deg)) if len(deg) else 0.0,
        "graph_leaf_fraction":float(np.mean(deg<=1)) if len(deg) else 0.0,
        "graph_density":float(density),
        "graph_n_components":int(len(comps)),
        "graph_diameter":int(diameter),
        "graph_cycle_rank":int(cycle_rank),
        "graph_mean_clustering":float(np.mean(local)) if local else 0.0,
    }


def _cell_shape_features(cells, E, C):
    cset=set(int(x) for x in cells)
    C2=C[C.cell_label.astype(int).isin(cset)].copy()
    area=dict(zip(C2.cell_label.astype(int), C2.pixels.astype(float)))
    per=defaultdict(float)
    for r in E.itertuples():
        a=int(r.cell_i)
        if a in cset: per[a]+=float(r.length)
        if r.kind=="cell_cell":
            b=int(r.cell_j)
            if b in cset: per[b]+=float(r.length)

    compact=[]; elong_proxy=[]
    for c in cset:
        A=float(area.get(c,np.nan)); P=float(per.get(c,np.nan))
        if np.isfinite(A) and np.isfinite(P) and P>0:
            compact.append(4*math.pi*A/(P*P))
            elong_proxy.append(P/max(math.sqrt(A),1e-12))
    areas=np.array(list(area.values()),float)
    return {
        "cell_area_mean":float(np.mean(areas)) if len(areas) else np.nan,
        "cell_area_cv":_safe_cv(areas),
        "cell_compactness_mean":float(np.mean(compact)) if compact else np.nan,
        "cell_compactness_cv":_safe_cv(compact),
        "cell_perimeter_sqrt_area_mean":float(np.mean(elong_proxy)) if elong_proxy else np.nan,
        "cell_perimeter_sqrt_area_cv":_safe_cv(elong_proxy),
    }


def _interface_features(cells,E):
    cset=set(int(x) for x in cells)
    q=E[
        E.cell_i.astype(int).isin(cset) &
        ((E.kind=="cell_boundary") | E.cell_j.fillna(-1).astype(int).isin(cset))
    ].copy()
    if len(q)==0:
        return {}
    cur=np.abs(pd.to_numeric(q.curvature,errors="coerce").to_numpy(float))
    conf=pd.to_numeric(q.curvature_confidence,errors="coerce").to_numpy(float)
    leng=pd.to_numeric(q.length,errors="coerce").to_numpy(float)
    orient=np.mod(np.arctan2(q.tangent_y.to_numpy(float),q.tangent_x.to_numpy(float)),math.pi)
    cellb=(q.kind=="cell_boundary").to_numpy()
    low=(cur<1e-3) | (conf<0.15)
    return {
        "n_interfaces":int(len(q)),
        "cell_boundary_interface_fraction":float(np.mean(cellb)),
        "cell_cell_interface_fraction":float(1-np.mean(cellb)),
        "interface_length_mean":float(np.nanmean(leng)),
        "interface_length_cv":_safe_cv(leng),
        "curvature_abs_mean":float(np.nanmean(cur)),
        "curvature_abs_median":float(np.nanmedian(cur)),
        "curvature_abs_q90":float(np.nanquantile(cur,.9)),
        "curvature_confidence_mean":float(np.nanmean(conf)),
        "low_information_curvature_fraction":float(np.nanmean(low)),
        "interface_orientation_entropy":_entropy(orient,bins=12,period=math.pi),
    }


def _junction_features(cells,E,J):
    cset=set(int(x) for x in cells)
    eids=set(
        E.loc[
            E.cell_i.astype(int).isin(cset) &
            ((E.kind=="cell_boundary") | E.cell_j.fillna(-1).astype(int).isin(cset)),
            "interface_id"
        ].astype(int)
    )
    counts=[]
    for r in J.itertuples():
        inc=[int(x) for x in r.incident_interfaces if int(x) in eids]
        if len(inc)>=3:
            counts.append(len(inc))
    n=len(counts)
    return {
        "n_junctions":int(n),
        "junctions_per_cell":float(n/max(len(cset),1)),
        "junction_incidence_mean":float(np.mean(counts)) if counts else 0.0,
        "junction_incidence_cv":_safe_cv(counts) if counts else 0.0,
        "high_order_junction_fraction":float(np.mean(np.asarray(counts)>3)) if counts else 0.0,
    }


def _gap_features(cells,E,B):
    cset=set(int(x) for x in cells)
    q=E[(E.kind=="cell_boundary") & E.cell_i.astype(int).isin(cset)]
    if len(q)==0:
        return {
            "n_background_components_touched":0,
            "n_internal_gaps_touched":0,
            "n_exterior_components_touched":0,
            "internal_gap_interface_fraction":0.0,
        }
    binfo={int(r.background_component):(str(r.kind),int(r.pixels))
           for r in B.itertuples()} if len(B) else {}
    comps=sorted(set(int(x) for x in q.background_component.dropna()))
    kinds=[binfo.get(x,("internal_gap",0))[0] for x in comps]
    internal={x for x in comps if binfo.get(x,("internal_gap",0))[0]=="internal_gap"}
    qint=q.background_component.fillna(-1).astype(int).isin(internal)
    pix=[binfo.get(x,("internal_gap",0))[1] for x in internal]
    return {
        "n_background_components_touched":int(len(comps)),
        "n_internal_gaps_touched":int(sum(k=="internal_gap" for k in kinds)),
        "n_exterior_components_touched":int(sum(k=="exterior" for k in kinds)),
        "internal_gap_interface_fraction":float(np.mean(qint)) if len(q) else 0.0,
        "internal_gap_pixels_median":float(np.median(pix)) if pix else 0.0,
        "internal_gap_pixels_q90":float(np.quantile(pix,.9)) if pix else 0.0,
    }


def patch_features(cells,E,B,J,C):
    out={"n_patch_cells":int(len(cells))}
    out.update(_graph_features(cells,E))
    out.update(_cell_shape_features(cells,E,C))
    out.update(_interface_features(cells,E))
    out.update(_junction_features(cells,E,J))
    out.update(_gap_features(cells,E,B))
    return out


def add_expression_features(df, source_cells, label_to_cell):
    """
    Optional. Looks for transcript-count-like columns in vendor cells metadata.
    Returns df unchanged if they are unavailable.
    """
    if source_cells is None or label_to_cell is None or len(source_cells)==0:
        return df
    candidates=[
        "transcript_counts","transcript_count","total_counts",
        "num_transcripts","n_transcripts"
    ]
    col=next((c for c in candidates if c in source_cells.columns),None)
    if col is None or "cell_id" not in source_cells.columns:
        return df
    m=dict(zip(source_cells.cell_id.astype(str),
               pd.to_numeric(source_cells[col],errors="coerce")))
    return df
