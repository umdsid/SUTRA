from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

from sutra.gap_audit.core import (
    GapAuditConfig,discover_morphology_files,
    polygons_from_boundary_table_generic,rasterize_polygons,
    internal_gap_metrics,load_morphology,compare_to_morphology
)


def _find(root,pattern):
    hits=list(Path(root).rglob(pattern))
    return hits[0] if hits else None


def _maybe_import_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def audit_sample(project,sample,cfg):
    sroot=project/"data"/sample
    cb=_find(sroot,"*cell_boundaries*.parquet")
    nb=_find(sroot,"*nucleus_boundaries*.parquet")
    morphs=discover_morphology_files(sroot)

    if cb is None:
        raise FileNotFoundError(f"{sample}: no cell boundaries")

    cdf=pd.read_parquet(cb)
    polys,bounds=polygons_from_boundary_table_generic(cdf)
    cell_mask,rmeta=rasterize_polygons(polys,bounds,cfg.raster_max_pixels)
    gapstats,filled,internal=internal_gap_metrics(cell_mask)

    report={
        "sample":sample,
        "cell_boundary_file":str(cb),
        "n_segmented_cells":int(len(polys)),
        "raster":rmeta,
        **gapstats,
        "morphology_files":[str(p) for p in morphs],
        "n_morphology_files":len(morphs),
    }

    if nb is not None:
        ndf=pd.read_parquet(nb)
        npolys,_=polygons_from_boundary_table_generic(ndf)
        nmask,_=rasterize_polygons(npolys,bounds,cfg.raster_max_pixels)
        report["nucleus_boundary_file"]=str(nb)
        report["n_segmented_nuclei"]=int(len(npolys))
        report["nucleus_coverage_fraction_of_cell_envelope"]=float(
            (nmask & filled).sum()/max(filled.sum(),1)
        )
    else:
        nmask=None
        report["nucleus_boundary_file"]=None

    out=project/"results"/"gap_morphology_audit"/sample
    out.mkdir(parents=True,exist_ok=True)

    plt=_maybe_import_plt()
    report["plotting_available"]=bool(plt is not None)

    if plt is not None:
        fig=plt.figure(figsize=(12,4))
        ax=fig.add_subplot(131)
        ax.imshow(cell_mask,origin="lower")
        ax.set_title("Cell segmentation"); ax.axis("off")
        ax=fig.add_subplot(132)
        ax.imshow(internal,origin="lower")
        ax.set_title("Internal unlabeled regions"); ax.axis("off")
        ax=fig.add_subplot(133)
        ax.imshow(filled,origin="lower")
        ax.set_title("Filled segmentation envelope"); ax.axis("off")
        fig.tight_layout()
        fig.savefig(out/"segmentation_gap_audit.png",dpi=180,bbox_inches="tight")
        plt.close(fig)

    if morphs:
        try:
            img,stride=load_morphology(morphs[0],cfg.morphology_downsample_max_dim)
            mstats,tissue_img,unassigned=compare_to_morphology(
                cell_mask,img,cfg.tissue_intensity_quantile
            )
            report["primary_morphology_file"]=str(morphs[0])
            report["morphology_stride"]=int(stride)
            report.update(mstats)

            if plt is not None:
                from scipy import ndimage
                cm=ndimage.zoom(
                    cell_mask.astype(np.uint8),
                    zoom=(img.shape[0]/cell_mask.shape[0],img.shape[1]/cell_mask.shape[1]),
                    order=0
                )[:img.shape[0],:img.shape[1]].astype(bool)

                fig=plt.figure(figsize=(14,5))
                ax=fig.add_subplot(131)
                ax.imshow(img,cmap="gray",origin="lower")
                ax.set_title("Morphology"); ax.axis("off")
                ax=fig.add_subplot(132)
                ax.imshow(img,cmap="gray",origin="lower")
                ax.contour(cm.astype(float),levels=[0.5],linewidths=0.5)
                ax.set_title("Morphology + cell boundaries"); ax.axis("off")
                ax=fig.add_subplot(133)
                ax.imshow(unassigned,origin="lower")
                ax.set_title("Morphology-supported but unassigned"); ax.axis("off")
                fig.tight_layout()
                fig.savefig(out/"morphology_gap_overlay.png",dpi=180,bbox_inches="tight")
                plt.close(fig)
        except Exception as e:
            report["morphology_analysis_error"]=f"{type(e).__name__}: {e}"

    (out/"gap_audit_summary.json").write_text(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    args=ap.parse_args()
    project=Path(args.project_root).resolve()
    cfg=GapAuditConfig()

    gb=json.loads(
        (project/"results"/"tranche2_1_gateB"/"gateB_certificate.json").read_text()
    )
    samples=[x["sample"] for x in gb["sample_certificates"]]

    outroot=project/"results"/"gap_morphology_audit"
    outroot.mkdir(parents=True,exist_ok=True)

    print("STRATA 0.4.10 | Gap + morphology integrity audit")
    reports=[]
    for i,sample in enumerate(samples,1):
        print(f"[{i}/{len(samples)}] {sample}",flush=True)
        r=audit_sample(project,sample,cfg)
        reports.append(r)
        print(
            f"    cells={r['n_segmented_cells']:,} "
            f"internal_gap={100*r['internal_gap_fraction_of_filled_envelope']:.2f}% "
            f"morphology_files={r['n_morphology_files']} "
            f"plots={'YES' if r['plotting_available'] else 'NO'}",
            flush=True
        )
        if "unassigned_fraction_of_morphology_tissue_support" in r:
            print(
                f"    morphology-supported unassigned="
                f"{100*r['unassigned_fraction_of_morphology_tissue_support']:.2f}%",
                flush=True
            )
        if "morphology_analysis_error" in r:
            print(f"    morphology analysis warning: {r['morphology_analysis_error']}",flush=True)

    overall={
        "strata_version":"0.4.10",
        "audit":"gap_morphology_integrity",
        "sample_reports":reports,
    }
    (outroot/"gap_morphology_audit.json").write_text(json.dumps(overall,indent=2))
    print()
    print(f"Audit summary: {outroot/'gap_morphology_audit.json'}")

if __name__=="__main__":
    main()
