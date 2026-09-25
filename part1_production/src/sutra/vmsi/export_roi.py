from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
from sutra.io.boundaries import read_boundary_table, polygons_from_boundary_table

def _find(root, pattern):
    hits=list(Path(root).rglob(pattern))
    if not hits: raise FileNotFoundError(f"{pattern} under {root}")
    return hits[0]

def _components(adj):
    seen=set(); comps=[]
    for seed in adj:
        if seed in seen: continue
        q=deque([seed]);seen.add(seed);comp=[]
        while q:
            u=q.popleft();comp.append(u)
            for v in adj.get(u,()):
                if v not in seen:
                    seen.add(v);q.append(v)
        comps.append(comp)
    return comps

def choose_connected_roi(edges,cells,target_cells):
    adj={}
    for r in edges.itertuples(index=False):
        a,b=str(r.cell_i),str(r.cell_j)
        adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
    if not adj:return []
    largest=max(_components(adj),key=len)
    lset=set(largest)
    c=cells.copy();c["cell_id"]=c.cell_id.astype(str)
    c=c[c.cell_id.isin(lset)].copy()
    cx=float(np.median(c.x_centroid));cy=float(np.median(c.y_centroid))
    c["degree"]=c.cell_id.map(lambda x:len(adj.get(str(x),())))
    qdeg=float(c.degree.quantile(.75))
    pool=c[c.degree>=qdeg]
    if pool.empty: pool=c
    d2=(pool.x_centroid-cx)**2+(pool.y_centroid-cy)**2
    seed=str(pool.loc[d2.idxmin(),"cell_id"])
    out=[];seen={seed};q=deque([seed])
    while q and len(out)<target_cells:
        u=q.popleft();out.append(u)
        for v in sorted(adj.get(u,()),key=lambda x:(-len(adj.get(x,())),x)):
            if v in lset and v not in seen:
                seen.add(v);q.append(v)
    return out

def export_sample(project,sample,target_cells,outroot):
    data=project/"data"/sample
    cells=pd.read_parquet(_find(data,"*cells.parquet"))
    cells["cell_id"]=cells.cell_id.astype(str)
    cert=pd.read_parquet(project/"results"/"tranche2_1_gateB"/sample/"certified_interfaces.parquet")
    rank={"marginal":0,"admissible":1,"high_confidence":2}
    cert=cert[cert.confidence_class.map(lambda x:rank.get(str(x),-1))>=1].copy()
    cert["cell_i"]=cert.cell_i.astype(str);cert["cell_j"]=cert.cell_j.astype(str)

    selected=choose_connected_roi(cert,cells,target_cells)
    sset=set(selected)
    bdf=read_boundary_table(_find(data,"*cell_boundaries*.parquet"))
    polygons,_=polygons_from_boundary_table(bdf)

    rows=[];mapping=[];cell_to_label={};label=1
    for cid in selected:
        g=polygons.get(cid)
        if g is None or g.is_empty: continue
        try: xy=np.asarray(g.exterior.coords,float)
        except Exception: continue
        if len(xy)<4: continue
        cell_to_label[cid]=label
        for order,(x,y) in enumerate(xy):
            rows.append({"label":label,"cell_id":cid,"vertex_order":order,"x":float(x),"y":float(y)})
        mapping.append({"label":label,"cell_id":cid})
        label+=1

    poly=pd.DataFrame(rows);mapping=pd.DataFrame(mapping)
    if poly.empty: raise RuntimeError(f"{sample}: empty ROI export")

    erows=[]
    for r in cert.itertuples(index=False):
        a,b=str(r.cell_i),str(r.cell_j)
        if a in cell_to_label and b in cell_to_label:
            erows.append({
                "label_i":cell_to_label[a],"label_j":cell_to_label[b],
                "cell_i":a,"cell_j":b,
                "confidence_class":str(r.confidence_class)
            })
    roi_edges=pd.DataFrame(erows)

    out=outroot/sample;out.mkdir(parents=True,exist_ok=True)
    poly.to_parquet(out/"roi_polygons.parquet",index=False)
    poly.to_csv(out/"roi_polygons.csv",index=False)
    mapping.to_csv(out/"label_to_cell.csv",index=False)
    roi_edges.to_csv(out/"roi_gateB_edges.csv",index=False)

    adj={}
    for r in cert.itertuples(index=False):
        a,b=str(r.cell_i),str(r.cell_j)
        adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
    comps=_components(adj)
    meta={
        "sample":sample,"requested_cells":target_cells,"exported_cells":int(len(mapping)),
        "largest_gateB_component_cells":int(max(map(len,comps)) if comps else 0),
        "n_gateB_components":int(len(comps)),
        "n_roi_gateB_edges":int(len(roi_edges)),
        "bounds":[float(poly.x.min()),float(poly.y.min()),float(poly.x.max()),float(poly.y.max())],
        "exchange_format":"CSV for legacy VMSI environment",
    }
    (out/"roi_export.json").write_text(json.dumps(meta,indent=2))
    return meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--target-cells",type=int,default=300)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    gb=json.loads((project/"results"/"tranche2_1_gateB"/"gateB_certificate.json").read_text())
    outroot=project/"results"/"tranche3_7b_vmsi_benchmark"/"exchange"
    outroot.mkdir(parents=True,exist_ok=True)
    print("STRATA 0.4.7 | VMSI benchmark ROI export")
    for sc in gb["sample_certificates"]:
        m=export_sample(project,sc["sample"],a.target_cells,outroot)
        print(f"  {m['sample']}: {m['exported_cells']} cells, {m['n_roi_gateB_edges']} Gate-B edges")
if __name__=="__main__": main()
