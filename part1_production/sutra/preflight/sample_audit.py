from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import pandas as pd

from .core import *


def audit_sample(project_s: str, sample: str, cfg_dict: dict):
    project=Path(project_s)
    cfg=PreflightConfig(**cfg_dict)
    t0=time.time()
    sroot=project/"data"/sample

    files={
        "matrix":find_one(sroot,["*cell_feature_matrix*.h5","*feature_matrix*.h5"]),
        "cells":find_one(sroot,"*cells.parquet"),
        "cell_boundaries":find_one(sroot,"*cell_boundaries*.parquet"),
        "nucleus_boundaries":find_one(sroot,"*nucleus_boundaries*.parquet"),
        "transcripts":find_one(sroot,["*transcripts*.parquet","*transcripts*.parquet.gz"]),
    }
    morphs=all_hits(sroot,[
        "*morphology*.ome.tif","*morphology*.ome.tiff",
        "*morphology*.tif","*morphology*.tiff","*.ome.tif","*.ome.tiff"
    ])

    report={
        "sample":sample,
        "files":{k:(str(v) if v else None) for k,v in files.items()},
        "morphology_files":[str(x) for x in morphs],
    }

    # File presence.
    required=("matrix","cells","cell_boundaries")
    missing=[k for k in required if files[k] is None]
    report["file_presence"]={
        "required_missing":missing,
        "status":"PASS" if not missing else "FAIL"
    }
    if missing:
        report["runtime_seconds"]=time.time()-t0
        return report

    # Matrix.
    mat=h5_feature_matrix_summary(files["matrix"])
    report["matrix"]= {
        k:v for k,v in mat.items() if k not in ("barcodes","feature_names","feature_ids")
    }
    report["_barcodes"]=mat["barcodes"]
    report["_feature_names"]=mat["feature_names"]
    report["_feature_ids"]=mat["feature_ids"]

    # Cells.
    cells_meta=parquet_metadata(files["cells"])
    cells=read_parquet_columns(
        files["cells"],
        ["cell_id","x_centroid","y_centroid","x","y","cell_area","nucleus_area"]
    )
    cell_ids=table_id_audit(cells,"cell_id")
    ccoord=coordinate_summary(cells)
    report["cells"]={
        "metadata":cells_meta,
        "id_audit":cell_ids,
        "coordinates":ccoord,
    }

    # Boundary tables.
    bmeta=parquet_metadata(files["cell_boundaries"])
    bdf=read_parquet_columns(
        files["cell_boundaries"],
        ["cell_id","vertex_x","vertex_y","x","y","vertex_order","vertex_index","vertex_id"]
    )
    bid=table_id_audit(bdf,"cell_id")
    bcoord=coordinate_summary(bdf)
    report["cell_boundaries"]={
        "metadata":bmeta,
        "id_audit":bid,
        "coordinates":bcoord,
        "bbox_overlap_with_cells":bbox_overlap_fraction(ccoord,bcoord),
    }

    if files["nucleus_boundaries"]:
        nmeta=parquet_metadata(files["nucleus_boundaries"])
        ndf=read_parquet_columns(
            files["nucleus_boundaries"],
            ["cell_id","vertex_x","vertex_y","x","y","vertex_order","vertex_index","vertex_id"]
        )
        ncoord=coordinate_summary(ndf)
        report["nucleus_boundaries"]={
            "metadata":nmeta,
            "id_audit":table_id_audit(ndf,"cell_id"),
            "coordinates":ncoord,
            "bbox_overlap_with_cells":bbox_overlap_fraction(ccoord,ncoord),
        }
    else:
        ndf=None
        report["nucleus_boundaries"]={"available":False}

    # ID overlap.
    cset=set(cells.cell_id.astype(str)) if "cell_id" in cells else set()
    bset=set(bdf.cell_id.astype(str)) if "cell_id" in bdf else set()
    barcodes=set(mat["barcodes"])
    report["id_overlap"]={
        "cells_vs_boundary_fraction":float(len(cset&bset)/max(len(cset),1)),
        "matrix_barcodes_vs_cells_fraction":float(len(barcodes&cset)/max(len(barcodes),1)) if barcodes else None,
        "cells_vs_matrix_barcodes_fraction":float(len(barcodes&cset)/max(len(cset),1)) if barcodes else None,
    }

    # Transcript stream.
    if files["transcripts"]:
        report["transcripts"]=transcript_stream_summary(files["transcripts"],ccoord,cfg)
    else:
        report["transcripts"]={"available":False}

    # Morphology metadata.
    msum=[]
    for p in morphs[:3]:
        try:
            msum.append(morphology_file_summary(p))
        except Exception as e:
            msum.append({"path":str(p),"readable":False,"error":f"{type(e).__name__}: {e}"})
    report["morphology"]=msum

    # Geometry coverage.
    polys=polygons_from_boundary_table(bdf)
    mask,rmeta=rasterize_union(polys,cfg.gap_raster_max_pixels)
    gm=gap_metrics(mask)
    report["geometry_coverage"]={**gm,"raster":rmeta}

    # Existing Gate-B topology.
    epath=project/"results"/"tranche2_1_gateB"/sample/"certified_interfaces.parquet"
    if epath.exists():
        edf=pd.read_parquet(epath)
        # production contact domain
        if "confidence_class" in edf.columns:
            use=edf[edf.confidence_class.astype(str).isin(["admissible","high_confidence"])].copy()
        else:
            use=edf
        es=edge_set_from_df(use)
        gstats=graph_components(es,sorted(cset))
        report["gateB_topology"]={**gstats,"n_edges":len(es)}
        # jitter surrogate
        eps=None
        cert=project/"results"/"tranche2_1_gateB"/"gateB_certificate.json"
        if cert.exists():
            try:
                eps=float(json.loads(cert.read_text()).get("frozen_contact_epsilon"))
            except Exception:
                eps=None
        report["gateB_topology"]["frozen_epsilon"]=eps
        report["gateB_topology"]["jitter_surrogate"]=topology_jitter_audit(cells,use,eps,cfg)
    else:
        report["gateB_topology"]={"available":False}

    # Geometry reconstruction availability.
    recon=project/"results"/"tranche2_2_mechanics_geometry"/sample/"mechanics_geometry_summary.json"
    if recon.exists():
        try:
            report["geometry_reconstruction"]=json.loads(recon.read_text())
        except Exception as e:
            report["geometry_reconstruction"]={"available":False,"error":str(e)}
    else:
        report["geometry_reconstruction"]={
            "available":False,
            "note":"Tranche 2.2 has not been run; preflight reports feasibility but does not require it."
        }

    # Per-sample status blocks.
    duplicate_fraction=cell_ids.get("duplicate_fraction",1.0)
    finite_fraction=ccoord.get("finite_fraction",0.0)
    cell_boundary_overlap=report["id_overlap"]["cells_vs_boundary_fraction"]
    matrix_overlap=report["id_overlap"]["matrix_barcodes_vs_cells_fraction"]

    data_fail=(
        duplicate_fraction>cfg.max_duplicate_cell_id_fraction or
        (1-finite_fraction)>cfg.max_nonfinite_coordinate_fraction or
        cell_boundary_overlap<cfg.min_boundary_cell_overlap_fraction or
        (matrix_overlap is not None and matrix_overlap<cfg.min_matrix_cell_overlap_fraction)
    )
    report["data_integrity_status"]="FAIL" if data_fail else "PASS"

    top=report["gateB_topology"]
    warn_struct=False
    if top.get("available",True) is False:
        warn_struct=True
    elif "isolated_fraction" in top:
        warn_struct=(
            top["isolated_fraction"]>cfg.max_isolated_fraction_warn or
            top["largest_component_fraction"]<cfg.min_largest_component_fraction_warn
        )
    if gm.get("available") and gm["internal_unassigned_fraction"]>cfg.max_internal_unassigned_fraction_warn:
        warn_struct=True
    report["structural_status"]="WARN" if warn_struct else "PASS"

    jitter=top.get("jitter_surrogate",{}) if isinstance(top,dict) else {}
    if jitter.get("available"):
        num_warn=jitter["min_candidate_jaccard_under_jitter"]<cfg.min_topology_jaccard_warn
        report["numerical_robustness_status"]="WARN" if num_warn else "PASS"
    else:
        report["numerical_robustness_status"]="WARN"

    report["runtime_seconds"]=float(time.time()-t0)
    return report
