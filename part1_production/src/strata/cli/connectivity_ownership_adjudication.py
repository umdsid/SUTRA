
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from strata.geometry_benchmark.baselines import polygons_from_boundaries, make_raster
from strata.connectivity.ownership_adjudication import (
    OwnershipConfig, adjudicate_ownership, audit_disconnected, topology_gate
)
from strata.production_geometry.materialize_radius5 import adjacency_from_labels, component_stats


SAMPLES = [
    "alzheimers",
    "gbm_reference_addon",
    "healthy_reference",
    "nondiseased_kidney",
    "prcc",
]


def _backup_once(p: Path):
    if not p.exists():
        return
    q = p.with_name(p.name + ".pre_ownership_v0513")
    if not q.exists():
        shutil.copy2(p, q)


def one(project_s: str, sample: str):
    project = Path(project_s)
    out = project/"results"/"production_geometry_r5_connectivity"/sample
    zpath = out/"production_geometry_connectivity.npz"
    tpath = out/"production_segmentation_connectivity.tif"
    rpath = out/"connectivity_raster_repair.json"

    z = np.load(zpath)
    labels = z["labels"].astype(np.int32)
    frozen_observed = z["observed_labels"].astype(np.int32)

    bpath = project/"results"/"tranche2_2b_geometry_benchmark"/sample/"primary_cell_boundaries.parquet"
    cpath = project/"results"/"tranche2_2b_geometry_benchmark"/sample/"primary_cells.csv"
    bdf = pd.read_parquet(bpath); bdf["cell_id"] = bdf.cell_id.astype(str)
    cells = pd.read_csv(cpath); cells["cell_id"] = cells.cell_id.astype(str)
    polys = polygons_from_boundaries(bdf)

    observed, meta = make_raster(polys, cells)
    if observed.shape != frozen_observed.shape or not np.array_equal(observed, frozen_observed):
        raise RuntimeError(f"{sample}: regenerated observed raster differs from frozen source")

    # Map continuous cell polygons into the mechanics-label namespace.
    polygons_by_label = {}
    for cid, poly in polys.items():
        lab = meta["cell_to_label"].get(str(cid))
        if lab is not None and poly is not None:
            polygons_by_label[int(lab)] = poly

    pre = audit_disconnected(labels)
    repaired, audit = adjudicate_ownership(
        labels, polygons_by_label, meta, OwnershipConfig()
    )
    post = audit_disconnected(repaired)

    cross = pd.DataFrame({
        "mechanics_label": np.arange(1, int(repaired.max())+1, dtype=np.int32),
        "cell_id": [meta["label_to_cell"][i] for i in range(1, int(repaired.max())+1)]
    })
    ids = dict(zip(cross.mechanics_label.astype(int), cross.cell_id.astype(str)))

    pairs = adjacency_from_labels(repaired)
    pred = {tuple(sorted((ids[a], ids[b]))) for a,b in pairs}
    e = pd.read_csv(
        project/"results"/"tranche2_2b_geometry_benchmark"/sample/"primary_gateB_edges.csv"
    )
    truth = {tuple(sorted((str(a),str(b)))) for a,b in zip(e.cell_i,e.cell_j)}
    tp = len(pred & truth)
    rec = tp/max(len(truth),1)
    new = len(pred-truth)/max(len(pred),1)
    comps = component_stats(pairs, len(ids))

    # Whole-specimen connectedness is descriptive. Multiple real tissue islands
    # are valid and are not bridged by this stage.
    pass_gate = topology_gate(
        post_disconnected_labels=len(post),
        gateB_contact_recall=rec,
        new_contact_burden=new,
    )

    report = {
        "sample": sample,
        "pre_ownership_disconnected_labels": len(pre),
        **audit,
        "post_disconnected_labels": len(post),
        "gateB_contact_recall": float(rec),
        "new_contact_burden": float(new),
        **comps,
        "largest_component_fraction_is_diagnostic_only": True,
        "cross_tissue_component_bridging_performed": False,
        "production_gate_pass": bool(pass_gate),
    }

    _backup_once(zpath)
    _backup_once(tpath)
    _backup_once(rpath)

    np.savez_compressed(
        zpath, labels=repaired, observed_labels=frozen_observed
    )
    tifffile.imwrite(tpath, repaired.astype(np.uint32), compression="zlib")
    cross.to_parquet(out/"label_to_cell_id.parquet", index=False)
    rpath.write_text(json.dumps(report, indent=2))

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    a = ap.parse_args()
    project = Path(a.project_root).resolve()

    reports = []
    print("STRATA 0.5.13 | Topology-safe raster ownership adjudication")
    print("Continuous Xenium polygons are authoritative for ambiguous pixel ownership.")
    print("No cross-tissue-component bridges. LCC is diagnostic, not an existence gate.")
    print()

    # Run serially: this stage touches local Shapely geometry and shared result files.
    for sample in SAMPLES:
        r = one(str(project), sample)
        reports.append(r)
        print(
            f"[DONE] {sample}: "
            f"pre={r['pre_ownership_disconnected_labels']} "
            f"reassigned={r['pixels_reassigned_total']} "
            f"contested={r['contested_pixels_reassigned']} "
            f"donors={r['donor_labels_touched']} "
            f"post={r['post_disconnected_labels']} "
            f"recall={100*r['gateB_contact_recall']:.2f}% "
            f"new={100*r['new_contact_burden']:.2f}% "
            f"LCC={100*r['largest_component_fraction']:.2f}%[diagnostic] "
            f"pass={r['production_gate_pass']}"
        )

    cert = {
        "strata_version": "0.5.13",
        "method": "topology_safe_polygon_ownership_adjudication",
        "scientific_contract": {
            "individual_mechanics_cells_must_be_4_connected": True,
            "multiple_specimen_tissue_components_allowed": True,
            "largest_component_fraction_is_gate": False,
            "cross_component_bridging_allowed": False,
            "unambiguous_foreign_pixel_reassignment_allowed": False,
            "donor_connectivity_may_worsen": False,
        },
        "sample_reports": reports,
        "gate_status": "PASS" if all(r["production_gate_pass"] for r in reports) else "HOLD",
    }
    gout = project/"results"/"production_geometry_r5_connectivity"
    cpath = gout/"connectivity_raster_certificate.json"
    _backup_once(cpath)
    cpath.write_text(json.dumps(cert, indent=2))

    print()
    print("TOPOLOGY-SAFE OWNERSHIP GATE:", cert["gate_status"])
    print("Certificate:", cpath)
    if cert["gate_status"] != "PASS":
        raise SystemExit(5)


if __name__ == "__main__":
    main()
