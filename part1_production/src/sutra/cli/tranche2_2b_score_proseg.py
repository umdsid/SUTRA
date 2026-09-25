from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

from sutra.geometry_benchmark.proseg_io import proseg_polygons,match_proseg_to_original


def _read_table(path_stem: Path):
    candidates=[
        path_stem.with_suffix(".csv.gz"),
        path_stem.with_suffix(".parquet"),
        path_stem.with_suffix(".csv"),
    ]
    for p in candidates:
        if p.exists():
            if str(p).endswith(".parquet"):
                return pd.read_parquet(p),str(p)
            return pd.read_csv(p),str(p)
    return None,None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--sample",required=True)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    out=project/"results"/"tranche2_2b_geometry_benchmark"/a.sample
    polyfile=out/"proseg"/"cell_polygons.geojson.gz"
    if not polyfile.exists():
        raise SystemExit(f"missing Proseg polygons: {polyfile}")

    polys=proseg_polygons(polyfile)
    cells=pd.read_csv(out/"primary_cells.csv")
    cells["cell_id"]=cells.cell_id.astype(str)
    mapping=match_proseg_to_original(polys,cells)

    edges=pd.read_csv(out/"primary_gateB_edges.csv")
    truth={tuple(sorted((str(a),str(b)))) for a,b in zip(edges.cell_i,edges.cell_j)}

    from shapely.strtree import STRtree
    geoms=[r["geometry"] for r in polys]
    tree=STRtree(geoms)
    pred=set()
    for i,g in enumerate(geoms):
        for j in tree.query(g.buffer(1e-6)):
            j=int(j)
            if j<=i: continue
            h=geoms[j]
            if g.touches(h) or g.distance(h)<=1e-6:
                if i in mapping and j in mapping:
                    pred.add(tuple(sorted((mapping[i],mapping[j]))))

    tp=len(pred&truth)
    precision=tp/max(len(pred),1)
    recall=tp/max(len(truth),1)

    cdict=cells.set_index("cell_id")[["x_centroid","y_centroid"]]
    disp=[]
    for i,cid in mapping.items():
        if cid not in cdict.index: continue
        p=geoms[i]
        old=cdict.loc[cid].to_numpy(float)
        new=np.array([p.centroid.x,p.centroid.y])
        disp.append(float(np.linalg.norm(new-old)))

    txstats={}
    t,tsource=_read_table(out/"proseg"/"transcript_metadata")
    if t is not None:
        candidates=[
            c for c in t.columns
            if "cell" in c.lower() and ("assign" in c.lower() or c.lower()=="cell")
        ]
        # Proseg often calls this simply "assignment".
        candidates += [c for c in t.columns if c.lower()=="assignment" and c not in candidates]
        txstats["source"]=tsource
        txstats["columns"]=list(t.columns)
        if candidates:
            c=candidates[0]
            s=t[c]
            txstats["assignment_column"]=c
            txstats["assigned_fraction"]=float(
                (s.notna() &
                 (s.astype(str)!="UNASSIGNED") &
                 (s.astype(str)!="4294967295") &
                 (s.astype(str)!="0")).mean()
            )

    score={
        "method":"proseg",
        "n_proseg_cells":len(polys),
        "n_original_primary_cells":len(cells),
        "matched_fraction":len(mapping)/max(len(cells),1),
        "contact_precision_vs_gateB":float(precision),
        "contact_recall_vs_gateB":float(recall),
        "contact_f1_vs_gateB":float(2*precision*recall/max(precision+recall,1e-15)),
        "new_contact_burden":float(len(pred-truth)/max(len(pred),1)),
        "lost_gateB_fraction":float(len(truth-pred)/max(len(truth),1)),
        "median_centroid_displacement":float(np.median(disp)) if disp else None,
        "q95_centroid_displacement":float(np.quantile(disp,.95)) if disp else None,
        "transcript_assignment":txstats,
    }
    (out/"proseg_score.json").write_text(json.dumps(score,indent=2))
    print(json.dumps(score,indent=2))

if __name__=="__main__":main()
