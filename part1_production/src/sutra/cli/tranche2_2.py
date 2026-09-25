from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage

from sutra.geometry_reconstruction.reconstruct import (
    ReconstructionConfig, polygons_from_boundary_table,
    morphology_support_from_image, nucleus_support_from_boundaries,
    transcript_density_mask, reconstruct_partition, labels_to_boundary_table
)


def _find(root,patterns):
    if isinstance(patterns,str):
        patterns=[patterns]
    for pat in patterns:
        hits=list(Path(root).rglob(pat))
        if hits:
            return hits[0]
    return None


def _read_tiff(path):
    import tifffile
    return tifffile.imread(path)


def run_sample(project,sample,cfg):
    sroot=project/"data"/sample

    cb=_find(sroot,"*cell_boundaries*.parquet")
    nb=_find(sroot,"*nucleus_boundaries*.parquet")
    tx=_find(sroot,["*transcripts*.parquet","*transcripts*.parquet.gz"])
    morph=_find(sroot,[
        "*morphology*.ome.tif","*morphology*.tif",
        "*.ome.tif","*.ome.tiff"
    ])

    if cb is None:
        raise FileNotFoundError(f"{sample}: missing cell boundaries")

    cdf=pd.read_parquet(cb)
    cell_polys=polygons_from_boundary_table(cdf)

    cert=pd.read_parquet(
        project/"results"/"tranche2_1_gateB"/sample/"certified_interfaces.parquet"
    )
    cert=cert[cert.confidence_class.astype(str).isin(["admissible","high_confidence"])].copy()

    # Build base raster first to determine target shape.
    from sutra.geometry_reconstruction.reconstruct import _raster_geometry
    base_labels,meta=_raster_geometry(cell_polys,cfg.raster_max_pixels)

    morphology_support=None
    morphology_threshold=None
    if morph is not None:
        img=_read_tiff(morph)
        morphology_support,morphology_threshold=morphology_support_from_image(
            img,base_labels.shape,cfg.morphology_quantile
        )

    nucleus_support=None
    if nb is not None:
        ndf=pd.read_parquet(nb)
        npolys=polygons_from_boundary_table(ndf)
        nucleus_support=nucleus_support_from_boundaries(npolys,meta)

    tx_density=None
    if tx is not None:
        tdf=pd.read_parquet(tx)
        tx_density=transcript_density_mask(tdf,meta)

    projected,observed,evidence,meta,stats=reconstruct_partition(
        cell_polys,cert,morphology_support,nucleus_support,tx_density,cfg
    )

    out=project/"results"/"tranche2_2_mechanics_geometry"/sample
    out.mkdir(parents=True,exist_ok=True)

    np.savez_compressed(
        out/"mechanics_geometry_raster.npz",
        observed_labels=observed.astype(np.int32),
        reconstructed_labels=projected.astype(np.int32),
        evidence=evidence.astype(np.float32),
        morphology_support=(morphology_support.astype(np.uint8) if morphology_support is not None else np.zeros_like(projected,dtype=np.uint8)),
        nucleus_support=(nucleus_support.astype(np.uint8) if nucleus_support is not None else np.zeros_like(projected,dtype=np.uint8)),
        transcript_density=(tx_density.astype(np.float32) if tx_density is not None else np.zeros_like(projected,dtype=np.float32)),
    )

    # Try to materialize polygons if scikit-image exists in native env; raster remains authoritative.
    btab=labels_to_boundary_table(projected,meta)
    if len(btab):
        btab.to_parquet(out/"mechanics_cell_boundaries.parquet",index=False)

    report={
        "sample":sample,
        "cell_boundary_file":str(cb),
        "nucleus_boundary_file":str(nb) if nb else None,
        "transcript_file":str(tx) if tx else None,
        "morphology_file":str(morph) if morph else None,
        "n_cells":int(len(cell_polys)),
        "n_certified_edges":int(len(cert)),
        "morphology_threshold":morphology_threshold,
        "polygon_boundary_materialized":bool(len(btab)),
        **stats,
    }
    (out/"mechanics_geometry_summary.json").write_text(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    cfgj=json.loads((project/"configs"/"tranche2_2.json").read_text())
    cfg=ReconstructionConfig(
        max_assignment_distance=float(cfgj["max_assignment_distance"]),
        neighbor_candidate_hops=int(cfgj["neighbor_candidate_hops"]),
        raster_max_pixels=int(cfgj["raster_max_pixels"]),
        morphology_quantile=float(cfgj["morphology_quantile"]),
    )

    gb=json.loads(
        (project/"results"/"tranche2_1_gateB"/"gateB_certificate.json").read_text()
    )
    samples=[r["sample"] for r in gb["sample_certificates"]]

    outroot=project/"results"/"tranche2_2_mechanics_geometry"
    outroot.mkdir(parents=True,exist_ok=True)

    print("STRATA 0.5.0 | Tranche 2.2 | Mechanics-grade geometry reconstruction")
    reports=[]
    for i,s in enumerate(samples,1):
        print(f"[{i}/{len(samples)}] {s}",flush=True)
        r=run_sample(project,s,cfg)
        reports.append(r)
        print(
            f"    observed={100*r['observed_fraction_of_tissue']:.2f}% "
            f"candidate={100*r['candidate_unassigned_fraction_of_tissue']:.2f}% "
            f"reconstructed={100*r['accepted_reconstruction_fraction_of_tissue']:.2f}% "
            f"remaining={100*r['remaining_unassigned_fraction_of_tissue']:.2f}% "
            f"preserved={r['observed_pixels_preserved']}",
            flush=True
        )

    overall={
        "strata_version":"0.5.0",
        "tranche":"2.2",
        "contract":"observed Xenium polygons preserved as immutable cores; morphology/nucleus/transcript-supported unassigned tissue reconstructed under Gate-B topology",
        "sample_reports":reports,
        "run_status":"PASS" if all(r["observed_pixels_preserved"] for r in reports) else "FAIL",
        "note":"This tranche constructs a mechanics-grade geometry; it does not replace Level-0 measured geometry."
    }
    (outroot/"tranche2_2_certificate.json").write_text(json.dumps(overall,indent=2))
    print()
    print(f"Tranche 2.2 run: {overall['run_status']}")
    print(f"Certificate: {outroot/'tranche2_2_certificate.json'}")

if __name__=="__main__":
    main()
