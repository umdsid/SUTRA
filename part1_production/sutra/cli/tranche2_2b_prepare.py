from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

from strata.geometry_benchmark.components import ComponentConfig,select_primary_component
from strata.geometry_benchmark.baselines import (
    polygons_from_boundaries,make_raster,constrained_nearest_completion,
    transcript_density,transcript_watershed_completion,score_labels
)


def _find(root,patterns):
    if isinstance(patterns,str):patterns=[patterns]
    for pat in patterns:
        hits=list(Path(root).rglob(pat))
        if hits:return sorted(hits)[0]
    return None


def _unassigned_mask(series):
    s=series.astype(str)
    return series.isna() | s.isin({
        "UNASSIGNED","unassigned","0","0.0","None","nan","4294967295"
    })


def _decode_scalar(v):
    if isinstance(v,(bytes,np.bytes_)):
        return v.decode("utf-8")
    return v


def _normalize_xenium_csv(tx: pd.DataFrame):
    """
    Normalize only the benchmark exchange copy.

    Proseg's Xenium CSV reader parses text fields as CSV strings and numeric
    fields from text, so CSV avoids Arrow string-encoding incompatibilities.
    No original Xenium file is modified.
    """
    out=tx.copy()

    # Xenium preset-relevant string columns.
    for c in ("feature_name","cell_id","fov_name"):
        if c in out.columns:
            out[c]=out[c].map(_decode_scalar)
            # retain native missing values where possible
            out[c]=out[c].astype("string")

    # Canonical numeric types/values expected semantically by Xenium.
    for c in ("x_location","y_location","z_location","qv"):
        if c in out.columns:
            out[c]=pd.to_numeric(out[c],errors="coerce")
    if "transcript_id" in out.columns:
        out["transcript_id"]=pd.to_numeric(out["transcript_id"],errors="raise").astype("uint64")
    if "overlaps_nucleus" in out.columns:
        out["overlaps_nucleus"]=pd.to_numeric(out["overlaps_nucleus"],errors="raise").astype("uint8")

    return out


def _schema_report(df):
    rep={}
    for c in df.columns:
        s=df[c]
        sample=None
        for v in s.head(100):
            if pd.notna(v):
                sample=type(v).__name__
                break
        rep[c]={
            "pandas_dtype":str(s.dtype),
            "python_value_type_sample":sample,
            "null_fraction":float(s.isna().mean()),
        }
    return rep


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--sample",required=True)
    ap.add_argument("--max-cells",type=int,default=0)
    a=ap.parse_args()
    project=Path(a.project_root).resolve();sample=a.sample
    sroot=project/"data"/sample
    out=project/"results"/"tranche2_2b_geometry_benchmark"/sample
    out.mkdir(parents=True,exist_ok=True)

    cells=pd.read_parquet(_find(sroot,"*cells.parquet"))
    cells["cell_id"]=cells.cell_id.astype(str)
    primary,comp=select_primary_component(cells,ComponentConfig())

    pcells=cells[cells.cell_id.isin(primary)].copy()
    if a.max_cells and len(pcells)>a.max_cells:
        cx=float(pcells.x_centroid.median());cy=float(pcells.y_centroid.median())
        d2=(pcells.x_centroid-cx)**2+(pcells.y_centroid-cy)**2
        pcells=pcells.loc[d2.nsmallest(a.max_cells).index].copy()
        primary=set(pcells.cell_id.astype(str))
        comp["smoke_subset_cells"]=len(primary)

    bpath=_find(sroot,"*cell_boundaries*.parquet")
    bdf=pd.read_parquet(bpath)
    bdf["cell_id"]=bdf.cell_id.astype(str)
    bdf=bdf[bdf.cell_id.isin(primary)].copy()
    polys=polygons_from_boundaries(bdf)

    epath=project/"results"/"tranche2_1_gateB"/sample/"certified_interfaces.parquet"
    edf=pd.read_parquet(epath)
    edf["cell_i"]=edf.cell_i.astype(str);edf["cell_j"]=edf.cell_j.astype(str)
    if "confidence_class" in edf.columns:
        edf=edf[edf.confidence_class.astype(str).isin(["admissible","high_confidence"])]
    edf=edf[edf.cell_i.isin(primary)&edf.cell_j.isin(primary)].copy()
    edges=list(zip(edf.cell_i,edf.cell_j))

    observed,meta=make_raster(polys,pcells)
    nearest,nmeta=constrained_nearest_completion(observed,meta,edges,max_distance=8.0)

    txpath=_find(sroot,["*transcripts.parquet","*transcripts.parquet.gz"])
    tx=None;xy=np.empty((0,2),float)
    tx_filter_stats={}
    original_schema=None
    normalized_schema=None
    if txpath:
        tx=pd.read_parquet(txpath)
        original_schema=_schema_report(tx)

        xcol="x_location" if "x_location" in tx.columns else "x"
        ycol="y_location" if "y_location" in tx.columns else "y"
        xmin,ymin,xmax,ymax=meta["bounds"]
        spatial=(tx[xcol]>=xmin)&(tx[xcol]<=xmax)&(tx[ycol]>=ymin)&(tx[ycol]<=ymax)

        cell_col="cell_id" if "cell_id" in tx.columns else None
        if cell_col:
            assigned_selected=tx[cell_col].map(_decode_scalar).astype(str).isin(primary)
            unassigned=_unassigned_mask(tx[cell_col].map(_decode_scalar))
            keep=spatial & (assigned_selected | unassigned)
            tx_filter_stats={
                "spatial_rows":int(spatial.sum()),
                "assigned_to_selected_rows":int((spatial&assigned_selected).sum()),
                "unassigned_rows":int((spatial&unassigned).sum()),
                "excluded_assigned_other_cells_rows":int(
                    (spatial & ~(assigned_selected|unassigned)).sum()
                ),
            }
        else:
            keep=spatial
            tx_filter_stats={"spatial_rows":int(spatial.sum()),"cell_id_column":None}

        tx=tx.loc[keep].copy()
        xy=tx[[xcol,ycol]].to_numpy(float)

    methods={
        "constrained_nearest":{
            "metadata":nmeta,
            "score":score_labels(nearest,observed,meta,edges,xy)
        }
    }
    np.savez_compressed(out/"baseline_constrained_nearest.npz",
                        labels=nearest,observed=observed)

    if tx is not None:
        try:
            dens=transcript_density(tx,meta)
            water,wmeta=transcript_watershed_completion(observed,dens)
            methods["transcript_watershed"]={
                "metadata":wmeta,
                "score":score_labels(water,observed,meta,edges,xy)
            }
            np.savez_compressed(out/"baseline_transcript_watershed.npz",
                                labels=water,observed=observed,density=dens)
        except Exception as e:
            methods["transcript_watershed"]={
                "status":"SKIP","reason":f"{type(e).__name__}: {e}"
            }

    pcells.to_csv(out/"primary_cells.csv",index=False)
    bdf.to_parquet(out/"primary_cell_boundaries.parquet",index=False)
    edf.to_csv(out/"primary_gateB_edges.csv",index=False)

    if tx is not None:
        # Retain parquet for native STRATA diagnostics.
        tx.to_parquet(out/"primary_transcripts.parquet",index=False)

        # Proseg exchange: explicitly normalized CSV.gz.
        proseg_tx=_normalize_xenium_csv(tx)
        normalized_schema=_schema_report(proseg_tx)
        proseg_tx.to_csv(
            out/"primary_transcripts_proseg.csv.gz",
            index=False,
            compression="gzip",
        )

    report={
        "sample":sample,
        "primary_component":comp,
        "n_primary_cells":len(pcells),
        "n_primary_polygons":len(polys),
        "n_primary_gateB_edges":len(edf),
        "n_primary_transcripts":len(tx) if tx is not None else 0,
        "transcript_filter":tx_filter_stats,
        "transcript_source":str(txpath) if txpath else None,
        "proseg_exchange":"primary_transcripts_proseg.csv.gz" if tx is not None else None,
        "original_transcript_schema":original_schema,
        "normalized_proseg_schema":normalized_schema,
        "methods":methods,
        "raster_meta":{
            k:v for k,v in meta.items()
            if k not in ("cell_to_label","label_to_cell")
        },
    }
    (out/"prepare_summary.json").write_text(json.dumps(report,indent=2))

    print(json.dumps({
        "sample":sample,
        "primary_cells":len(pcells),
        "detached_cells":comp["detached_cells"],
        "plateau":comp["plateau_certified"],
        "nearest_F1":methods["constrained_nearest"]["score"]["contact_f1_vs_gateB"],
        "watershed_F1":methods.get("transcript_watershed",{}).get("score",{}).get("contact_f1_vs_gateB"),
        "proseg_input_transcripts":len(tx) if tx is not None else 0,
        "proseg_exchange":"CSV.GZ",
    },indent=2))

if __name__=="__main__":main()
