from __future__ import annotations
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
import tifffile

from sutra.connectivity.forensics import classify_cell
from sutra.connectivity.raster_repair import audit_disconnected

def _find(root,patterns):
    if isinstance(patterns,str): patterns=[patterns]
    for pat in patterns:
        hits=list(Path(root).rglob(pat))
        if hits: return sorted(hits)[0]
    return None

def one(project_s,sample):
    project=Path(project_s)
    mask=tifffile.imread(project/"results"/"production_geometry_r5_connectivity"/sample/"production_segmentation_connectivity.tif")
    bad_labels=audit_disconnected(mask)

    cross=pd.read_parquet(project/"results"/"production_geometry_r5_connectivity"/sample/"label_to_cell_id.parquet")
    lab_to_id=dict(zip(cross.mechanics_label.astype(int),cross.cell_id.astype(str)))
    bad_ids=[lab_to_id[x] for x in bad_labels]

    sroot=project/"data"/sample
    bdf=pd.read_parquet(_find(sroot,"*cell_boundaries*.parquet"))
    bdf["cell_id"]=bdf.cell_id.astype(str)
    groups={cid:g for cid,g in bdf[bdf.cell_id.isin(bad_ids)].groupby("cell_id",sort=False)}

    nuc={}
    npth=_find(sroot,["*nucleus_boundaries*.parquet","*nucleus_boundary*.parquet"])
    if npth:
        ndf=pd.read_parquet(npth); ndf["cell_id"]=ndf.cell_id.astype(str)
        xcol="vertex_x" if "vertex_x" in ndf.columns else "x"
        ycol="vertex_y" if "vertex_y" in ndf.columns else "y"
        for cid,g in ndf.groupby("cell_id",sort=False):
            nuc[str(cid)]=(float(g[xcol].mean()),float(g[ycol].mean()))

    rows=[]
    for lab,cid in zip(bad_labels,bad_ids):
        g=groups.get(cid)
        if g is None:
            rows.append({"mechanics_label":lab,"cell_id":cid,"classification":"missing_boundary"})
        else:
            rows.append({"mechanics_label":lab,"cell_id":cid,**classify_cell(g,nuc.get(cid))})

    out=project/"results"/"polygon_connectivity_forensics"/sample
    out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/"failed_label_forensics.csv",index=False)
    counts={str(k):int(v) for k,v in df["classification"].value_counts().to_dict().items()} if len(df) else {}
    summary={"sample":sample,"n_failed_labels":len(rows),"classification_counts":counts}
    if len(df) and "secondary_area_fraction" in df.columns:
        m=df[df["classification"]=="continuous_geometry_multipart"]
        summary["multipart_median_secondary_area_fraction"]=float(m["secondary_area_fraction"].median()) if len(m) else None
        summary["multipart_q95_secondary_area_fraction"]=float(m["secondary_area_fraction"].quantile(.95)) if len(m) else None
    (out/"forensics_summary.json").write_text(json.dumps(summary,indent=2))
    return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",default=".")
    a=ap.parse_args(); project=Path(a.project_root).resolve()
    samples=["alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"]
    rs=[]
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs={ex.submit(one,str(project),s):s for s in samples}
        for f in as_completed(futs):
            r=f.result(); rs.append(r)
            print(f"[DONE] {r['sample']}: failed={r['n_failed_labels']} classes={r['classification_counts']}")
    rs.sort(key=lambda x:samples.index(x["sample"]))
    out=project/"results"/"polygon_connectivity_forensics"; out.mkdir(parents=True,exist_ok=True)
    (out/"polygon_connectivity_forensics_certificate.json").write_text(json.dumps({"strata_version":"0.5.12","sample_reports":rs},indent=2))
    print(f"Certificate: {out/'polygon_connectivity_forensics_certificate.json'}")
if __name__=="__main__": main()
