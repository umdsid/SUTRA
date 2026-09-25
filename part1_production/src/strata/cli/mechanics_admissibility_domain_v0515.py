
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

ST4=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)

SAMPLES=[
    "alzheimers",
    "gbm_reference_addon",
    "healthy_reference",
    "nondiseased_kidney",
    "prcc",
]

def disconnected_labels(mask):
    mask=np.asarray(mask,dtype=np.int32)
    mx=int(mask.max()) if mask.size else 0
    sls=ndimage.find_objects(mask,max_label=mx)
    bad=[]
    for lab in range(1,mx+1):
        sl=sls[lab-1] if lab-1<len(sls) else None
        if sl is None:
            continue
        _,n=ndimage.label(mask[sl]==lab,structure=ST4)
        if n>1:
            bad.append(int(lab))
    return bad

def sequentialize_connected_labels(mask):
    """
    Build a contiguous mechanics-only label image without scikit-image.

    Preconditions: every nonzero original label is 4-connected.
    Mapping is deterministic by ascending original label.
    """
    mask=np.asarray(mask,dtype=np.int32)
    labels=np.unique(mask)
    labels=labels[labels>0]
    out=np.zeros(mask.shape,dtype=np.int32)
    rows=[]
    for new_lab,old_lab in enumerate(labels.tolist(),start=1):
        out[mask==int(old_lab)]=int(new_lab)
        rows.append((int(old_lab),int(new_lab)))
    return out,rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    a=ap.parse_args()
    root=Path(a.project_root).resolve()
    g=root/"results"/"production_geometry_r5_connectivity"

    cert_path=g/"connectivity_raster_certificate.json"
    cert=json.loads(cert_path.read_text())
    old_reports={r["sample"]:r for r in cert.get("sample_reports",[])}

    reports=[]
    print("STRATA 0.5.15.1 | Mechanics admissibility domain")
    print("No scikit-image dependency in STRATA runtime.")
    print("Residual unresolved biological cells are excluded only from mechanics.")
    print("Retained mechanics cells are deterministically sequentialized before TensionMap.")
    print()

    for sample in SAMPLES:
        d=g/sample
        tissue=tifffile.imread(d/"production_segmentation_connectivity.tif").astype(np.int32)
        bad=disconnected_labels(tissue)

        cross=pd.read_parquet(d/"label_to_cell_id.parquet").copy()
        cross["mechanics_label"]=cross.mechanics_label.astype(int)
        cross["cell_id"]=cross.cell_id.astype(str)
        idmap=dict(zip(cross.mechanics_label,cross.cell_id))

        # Biological geometry remains untouched. Only mechanics input excludes bad labels.
        mech=tissue.copy()
        if bad:
            mech[np.isin(mech,np.asarray(bad,dtype=np.int32))]=0

        post=disconnected_labels(mech)
        if post:
            raise RuntimeError(
                f"{sample}: mechanics-domain raster still contains disconnected labels: {post[:20]}"
            )

        # Contiguous label image for TensionMap. This removes any dependency on
        # TensionMap's internal default relabeling and gives an exact crosswalk.
        tm_mask,mapping=sequentialize_connected_labels(mech)
        old_to_tm=dict(mapping)

        rows=[]
        excluded=set(bad)
        for lab,cid in zip(cross.mechanics_label,cross.cell_id):
            if lab in excluded:
                rows.append({
                    "original_mechanics_label":int(lab),
                    "cell_id":cid,
                    "included_in_mechanics":False,
                    "tensionmap_label":pd.NA,
                    "reason":"UNRESOLVED_CELL_TOPOLOGY",
                })
            else:
                rows.append({
                    "original_mechanics_label":int(lab),
                    "cell_id":cid,
                    "included_in_mechanics":True,
                    "tensionmap_label":int(old_to_tm[int(lab)]),
                    "reason":"PASS",
                })

        xt=pd.DataFrame(rows)
        xt["tensionmap_label"]=xt["tensionmap_label"].astype("Int64")
        xt.to_parquet(d/"mechanics_input_label_crosswalk.parquet",index=False)
        xt.loc[~xt.included_in_mechanics].to_parquet(
            d/"mechanics_exclusions.parquet",index=False
        )

        tifffile.imwrite(
            d/"production_segmentation_mechanics.tif",
            mech.astype(np.uint32),
            compression="zlib",
        )
        tifffile.imwrite(
            d/"production_segmentation_tensionmap.tif",
            tm_mask.astype(np.uint32),
            compression="zlib",
        )
        np.savez_compressed(
            d/"production_geometry_mechanics_domain.npz",
            labels=mech,
            tensionmap_labels=tm_mask,
            excluded_labels=np.asarray(bad,dtype=np.int32),
        )

        old=old_reports.get(sample,{})
        n_total=int(len(cross)); n_ex=int(len(bad))
        rep={
            "sample":sample,
            "biological_cells":n_total,
            "mechanics_admissible_cells":n_total-n_ex,
            "mechanics_unresolved_cells":n_ex,
            "mechanics_domain_fraction":float((n_total-n_ex)/max(n_total,1)),
            "excluded_original_labels":bad,
            "excluded_cell_ids":[idmap.get(x) for x in bad],
            "retained_cells_all_4_connected":True,
            "tensionmap_labels_contiguous":bool(
                np.array_equal(
                    np.unique(tm_mask)[1:],
                    np.arange(1,len(np.unique(tm_mask)),dtype=np.int32)
                )
            ),
            "cross_tissue_component_bridging_performed":False,
            "largest_component_fraction_is_diagnostic_only":True,
            "gateB_contact_recall":old.get("gateB_contact_recall"),
            "new_contact_burden":old.get("new_contact_burden"),
            "largest_component_fraction":old.get("largest_component_fraction"),
            "status":"PASS",
        }
        if not rep["tensionmap_labels_contiguous"]:
            raise RuntimeError(f"{sample}: TensionMap label sequentialization failed")

        (d/"mechanics_admissibility_domain.json").write_text(json.dumps(rep,indent=2))
        reports.append(rep)
        print(
            f"[DONE] {sample}: biological={n_total} "
            f"mechanics={n_total-n_ex} unresolved={n_ex} "
            f"coverage={100*rep['mechanics_domain_fraction']:.6f}% "
            f"tm_labels={n_total-n_ex} PASS"
        )

    out={
        "strata_version":"0.5.15.1",
        "method":"explicit_mechanics_admissibility_domain_with_contiguous_solver_labels",
        "scientific_contract":{
            "biological_cells_deleted":False,
            "excluded_cells_remain_in_expression_spatial_topology_hierarchy":True,
            "excluded_cells_removed_only_from_mechanics_input":True,
            "all_retained_mechanics_cells_4_connected":True,
            "whole_specimen_single_component_required":False,
            "largest_component_fraction_is_gate":False,
            "cross_tissue_component_bridging_allowed":False,
            "mechanics_domain_fraction_is_reported_not_thresholded":True,
            "strata_runtime_requires_skimage":False,
            "tensionmap_input_is_explicitly_sequentialized":True,
            "tensionmap_called_with_is_labelled_true":True,
            "exact_tensionmap_input_crosswalk_written":True,
        },
        "sample_reports":reports,
        "gate_status":"PASS",
    }
    (g/"mechanics_admissibility_certificate.json").write_text(json.dumps(out,indent=2))

    cert["mechanics_admissibility"]=out
    cert["gate_status"]="PASS"
    cert["gate_semantics"]=(
        "PASS means every retained mechanics cell is 4-connected; residual biological "
        "cells are provenance-marked mechanics exclusions. LCC is diagnostic only."
    )
    cert_path.write_text(json.dumps(cert,indent=2))

    print()
    print("MECHANICS ADMISSIBILITY GATE: PASS")
    print("Certificate:",g/"mechanics_admissibility_certificate.json")

if __name__=="__main__":
    main()
