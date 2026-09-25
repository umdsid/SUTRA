from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sutra.io.discovery import discover_xenium_samples
from sutra.io.xenium import read_10x_h5_metadata, read_cells
from strata_native_mechanics.corrections import CorrectionConfig, corrected_objects
from strata_native_mechanics.patches import partition_core_cells
from sutra.hierarchy import GraphID, SampleID, PatchID, InterfaceID
from sutra.hierarchy.builders.cells import build_cells
from sutra.hierarchy.builders.interfaces import build_interfaces
from sutra.hierarchy.builders.patches import build_patches
from sutra.hierarchy.builders.graph import build_graph
from sutra.hierarchy.provenance import Provenance
from sutra.hierarchy.sample import Sample
from sutra.hierarchy.hashing import sha256_file


SAMPLES = ("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def _mechanics_gate(project: Path):
    p = _require(
        project / "results" / "hierarchy_mechanics_v0691" /
        "hierarchy_mechanics_certificate.json"
    )
    d = json.loads(p.read_text())
    if not d.get("HIERARCHY_READY", False):
        raise RuntimeError("v0.6.9.1 HIERARCHY_READY is not true")
    return p, d


def _sample_asset_map(project: Path):
    found = discover_xenium_samples(project / "data", require_cells=True)
    return {x.sample_id: x for x in found}


def _load_corrected_interfaces(project: Path, sample: str):
    cache = project / "results" / "native_mechanics_cache" / sample
    E0 = pd.read_parquet(_require(cache / "interfaces.parquet"))
    J0 = pd.read_json(_require(cache / "junctions.jsonl"), lines=True)
    if "incident_interfaces" in J0.columns:
        J0["incident_interfaces"] = J0["incident_interfaces"].apply(
            lambda x: list(x) if isinstance(x, (list, tuple, np.ndarray)) else []
        )
    P = pd.read_parquet(
        _require(
            project / "results" / "corrections" / "v064" / sample /
            "mechanical_boundary_persistence.parquet"
        )
    )
    E, _, _ = corrected_objects(
        E0, J0, P, CorrectionConfig(), "persistent_boundaries"
    )
    return E0, E


def _interface_membership_by_barcode(E, label_to_barcode):
    d: dict[str, list[int]] = {}
    for r in E.itertuples():
        eid = int(r.interface_id)
        a = label_to_barcode[int(r.cell_i)]
        d.setdefault(a, []).append(eid)
        if not pd.isna(r.cell_j):
            b = label_to_barcode[int(r.cell_j)]
            d.setdefault(b, []).append(eid)
    return {k: tuple(sorted(set(v))) for k, v in d.items()}


def _owner_maps(project: Path, sample: str):
    # Reconstruct the exact deterministic ownership rule used in v0.6.9 from
    # the source production table, not from observability.
    t = pd.read_parquet(
        _require(
            project / "results" / "production_mechanics_v069" / sample /
            "production_core_tensions.parquet"
        ),
        columns=["interface_id", "owner_patch_id", "owner_anchor_cell"],
    )
    if not t.interface_id.is_unique:
        raise RuntimeError(f"{sample}: duplicate owner interface IDs")
    return (
        dict(zip(t.interface_id.astype(int), t.owner_patch_id.astype(int))),
        dict(zip(t.interface_id.astype(int), t.owner_anchor_cell.astype(int))),
    )


def _write_tables(
    out: Path,
    barcodes: list[str],
    cells,
    interfaces,
    patches,
    graph,
    features: pd.DataFrame,
):
    out.mkdir(parents=True, exist_ok=True)

    ctab = pa.table({
        "cell_index": pa.array([int(c.id) for c in cells], type=pa.int64()),
        "cell_id": pa.array(barcodes),
        "patch_id": pa.array([int(c.patch_id) for c in cells], type=pa.int32()),
        "x": pa.array([c.centroid_x for c in cells], type=pa.float64()),
        "y": pa.array([c.centroid_y for c in cells], type=pa.float64()),
        "area": pa.array([c.area for c in cells], type=pa.float64()),
        "matrix_column": pa.array([c.gene_index for c in cells], type=pa.int64()),
        "n_mechanics_interfaces": pa.array(
            [len(c.interface_ids) for c in cells], type=pa.int32()
        ),
    })
    pq.write_table(ctab, out / "cells.parquet", compression="zstd")

    itab = pa.table({
        "interface_id": pa.array([int(e.id) for e in interfaces], type=pa.int64()),
        "owner_patch_id": pa.array([int(e.patch_id) for e in interfaces], type=pa.int32()),
        "cell_i_index": pa.array([int(e.cell_i) for e in interfaces], type=pa.int64()),
        "cell_j_index": pa.array(
            [None if e.cell_j is None else int(e.cell_j) for e in interfaces],
            type=pa.int64(),
        ),
        "owner_cell_index": pa.array([int(e.owner) for e in interfaces], type=pa.int64()),
        "length": pa.array([e.length for e in interfaces], type=pa.float64()),
        "curvature": pa.array([e.curvature for e in interfaces], type=pa.float64()),
        "tension": pa.array([e.tension for e in interfaces], type=pa.float64()),
        "delta_pressure": pa.array(
            [e.delta_pressure for e in interfaces], type=pa.float64()
        ),
        "flags": pa.array([e.flags for e in interfaces], type=pa.uint16()),
    })
    pq.write_table(itab, out / "interfaces.parquet", compression="zstd")

    ptab = pa.table({
        "patch_id": pa.array([int(p.id) for p in patches], type=pa.int32()),
        "n_cells": pa.array([p.n_cells for p in patches], type=pa.int32()),
        "n_interfaces": pa.array([p.n_interfaces for p in patches], type=pa.int32()),
    })
    pq.write_table(ptab, out / "patches.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(features, preserve_index=False),
                   out / "features.parquet", compression="zstd")

    np.savez(
        out / "graph_csr.npz",
        indptr=np.asarray(graph.indptr),
        indices=np.asarray(graph.indices),
        interface_ids=np.asarray(graph.interface_ids),
        node_ids=np.asarray([int(x) for x in graph.node_ids], dtype=np.int64),
    )


def one_sample(project_s: str, sample: str, sample_index: int):
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    project = Path(project_s)
    assets = _sample_asset_map(project)[sample]

    matrix = read_10x_h5_metadata(assets.matrix, audit_counts=False)
    measured = read_cells(assets.cells)
    barcodes = [str(x) for x in matrix.barcodes]

    label_map = pd.read_parquet(
        _require(
            project / "results" / "production_geometry_r5" / sample /
            "label_to_cell_id.parquet"
        )
    )
    label_map["cell_id"] = label_map.cell_id.astype(str)
    label_to_barcode = dict(
        zip(label_map.mechanics_label.astype(int), label_map.cell_id)
    )
    primary_barcodes = set(label_map.cell_id)

    E0, E = _load_corrected_interfaces(project, sample)
    owner_patch, owner_label = _owner_maps(project, sample)

    # Not every corrected geometry interface belongs to the mechanics contact
    # graph. Boundary-only cells/interfaces are legitimate Level-0 geometry and
    # are retained without mechanics ownership.
    geometry_interface_ids=set(E.interface_id.astype(int))
    mechanics_owned_ids=set(owner_patch)
    unknown_owned=sorted(mechanics_owned_ids-geometry_interface_ids)
    if unknown_owned:
        raise RuntimeError(
            f"{sample}: {len(unknown_owned)} mechanics-owned interfaces are absent "
            "from corrected geometry"
        )
    nonmechanics_interface_ids=tuple(
        sorted(geometry_interface_ids-mechanics_owned_ids)
    )

    membership = _interface_membership_by_barcode(E, label_to_barcode)
    barcode_to_patch = {}
    for eid, pid in owner_patch.items():
        lab = owner_label[eid]
        barcode_to_patch.setdefault(label_to_barcode[lab], int(pid))

    # Patch membership is reconstructed from the frozen partition itself,
    # which is deterministic on E0 and patch_size=300.
    core_labels = partition_core_cells(E0, patch_size=300)
    label_patch = {}
    for pid, labs in enumerate(core_labels):
        for lab in labs:
            label_patch[int(lab)] = pid
    barcode_to_patch = {
        label_to_barcode[lab]: pid for lab, pid in label_patch.items()
    }

    cells, barcode_to_cell_id = build_cells(
        barcodes, measured, barcode_to_patch, membership
    )

    mech = project / "results" / "hierarchy_mechanics_v0691" / sample
    T = pd.read_parquet(_require(mech / "certified_tensions.parquet"))
    DP = pd.read_parquet(_require(mech / "certified_pressure_contrasts.parquet"))

    interfaces = build_interfaces(
        E,
        label_to_barcode,
        barcode_to_cell_id,
        T,
        DP,
        owner_patch,
        owner_label,
    )

    # Any measured cell not assigned to a mechanics core is retained in the
    # non-mechanics Level-0 patch -1. This includes both cells outside the
    # primary mechanics component and primary cells with no cell-cell contact.
    core_barcodes=set(barcode_to_patch)
    retention_barcodes=[b for b in barcodes if b not in core_barcodes]
    retention_cells=tuple(barcode_to_cell_id[b] for b in retention_barcodes)

    graph_id = GraphID(sample_index)
    patches = build_patches(
        core_labels,
        label_to_barcode,
        barcode_to_cell_id,
        owner_patch,
        graph_id,
        retention_cell_ids=retention_cells,
        retention_interface_ids=tuple(
            InterfaceID(eid) for eid in nonmechanics_interface_ids
        ),
    )
    graph = build_graph(
        graph_id,
        tuple(c.id for c in cells),
        interfaces,
    )

    features = pd.DataFrame({
        "feature_id": matrix.feature_ids,
        "feature_name": matrix.feature_names,
        "feature_type": matrix.feature_types,
        "feature_index": np.arange(len(matrix.feature_ids), dtype=np.int64),
    })

    gate_path = (
        project / "results" / "hierarchy_mechanics_v0691" /
        "hierarchy_mechanics_certificate.json"
    )
    v68_path = (
        project / "results" / "corrections" / "v068" /
        "rowspace_observability_certificate.json"
    )
    geom_path = (
        project / "results" / "production_geometry_r5" / sample /
        "production_geometry_summary.json"
    )

    prov = Provenance(
        geometry_certificate_sha256=sha256_file(geom_path),
        observability_certificate_sha256=sha256_file(v68_path),
        mechanics_certificate_sha256=sha256_file(gate_path),
        hierarchy_version="0.7.0-step3",
    )
    sample_obj = Sample(
        id=SampleID(sample_index),
        name=sample,
        patch_ids=tuple(PatchID(int(p.id)) for p in patches),
        provenance=prov,
    )

    out = project / "results" / "hierarchy_level0_v070" / sample
    _write_tables(out, barcodes, cells, interfaces, patches, graph, features)

    expression_manifest = {
        "schema_version": "sutra.level0.expression_pointer.v1",
        "source_matrix": str(Path(assets.matrix).resolve()),
        "source_matrix_sha256": sha256_file(assets.matrix),
        "matrix_shape_features_by_cells": list(matrix.shape),
        "nnz": int(matrix.nnz),
        "cell_order": "cells.parquet matrix_column / original 10x barcode order",
        "feature_order": "features.parquet feature_index / original 10x feature order",
        "expression_copied": False,
        "reason": "Level-0 preserves measured expression without duplicating the source sparse matrix.",
    }
    (out / "expression_manifest.json").write_text(
        json.dumps(expression_manifest, indent=2) + "\n"
    )

    n_tau = sum(e.tension is not None for e in interfaces)
    n_dp = sum(e.delta_pressure is not None for e in interfaces)
    report = {
        "sample": sample,
        "status": "PASS",
        "n_measured_cells": len(cells),
        "n_primary_mechanics_cells": len(primary_barcodes),
        "n_retained_nonmechanics_cells": len(retention_cells),
        "n_nonmechanics_geometry_interfaces": len(nonmechanics_interface_ids),
        "n_features": len(features),
        "n_interfaces": len(interfaces),
        "n_cell_cell_edges": graph.n_undirected_edges,
        "n_patches": len(patches),
        "n_certified_tensions_attached": int(n_tau),
        "n_certified_pressure_contrasts_attached": int(n_dp),
        "n_certified_tensions_source": int(len(T)),
        "n_certified_pressure_contrasts_source": int(len(DP)),
        "all_tensions_accounted": bool(n_tau == len(T)),
        "all_pressure_contrasts_accounted": bool(n_dp == len(DP)),
        "all_measured_cells_retained": bool(len(cells) == matrix.shape[1]),
        "all_measured_features_retained": bool(len(features) == matrix.shape[0]),
        "provenance": {
            "geometry_certificate_sha256": prov.geometry_certificate_sha256,
            "observability_certificate_sha256": prov.observability_certificate_sha256,
            "mechanics_certificate_sha256": prov.mechanics_certificate_sha256,
        },
    }
    if not (
        report["all_tensions_accounted"]
        and report["all_pressure_contrasts_accounted"]
        and report["all_measured_cells_retained"]
        and report["all_measured_features_retained"]
    ):
        report["status"] = "FAIL"

    (out / "materialization_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    project = Path(a.project_root).resolve()
    _mechanics_gate(project)

    assets = _sample_asset_map(project)
    missing = [s for s in SAMPLES if s not in assets]
    if missing:
        raise SystemExit(f"ERROR: missing expected data samples: {missing}")

    print("STRATA 0.7.0 | Step 3 | Real Level-0 materialization")
    print("All measured cells/features retained; mechanics attached only where certified.")
    print(f"Parallel specimen workers: {min(a.workers, len(SAMPLES))}\n")

    reports = []
    with ProcessPoolExecutor(max_workers=min(a.workers, len(SAMPLES))) as ex:
        futs = {
            ex.submit(one_sample, str(project), s, i): s
            for i, s in enumerate(SAMPLES)
        }
        for f in as_completed(futs):
            r = f.result()
            reports.append(r)
            print(
                f"[DONE] {r['sample']}: "
                f"cells={r['n_measured_cells']:,} "
                f"primary={r['n_primary_mechanics_cells']:,} "
                f"retention_cells={r['n_retained_nonmechanics_cells']:,} "
                f"geom_only_ifaces={r['n_nonmechanics_geometry_interfaces']:,} "
                f"interfaces={r['n_interfaces']:,} "
                f"tau={r['n_certified_tensions_attached']:,} "
                f"dp={r['n_certified_pressure_contrasts_attached']:,} "
                f"{r['status']}",
                flush=True,
            )

    reports.sort(key=lambda x: SAMPLES.index(x["sample"]))
    gate = all(r["status"] == "PASS" for r in reports)
    outroot = project / "results" / "hierarchy_level0_v070"
    cert = {
        "strata_version": "0.7.0",
        "stage": "Step 3 real Level-0 materialization",
        "source_mechanics_hierarchy_ready": True,
        "all_measured_cells_retained": all(
            r["all_measured_cells_retained"] for r in reports
        ),
        "all_measured_features_retained": all(
            r["all_measured_features_retained"] for r in reports
        ),
        "mechanics_recomputed": False,
        "geometry_modified": False,
        "expression_modified": False,
        "sample_reports": reports,
        "STEP3_GATE": "PASS" if gate else "FAIL",
    }
    outroot.mkdir(parents=True, exist_ok=True)
    (outroot / "step3_certificate.json").write_text(
        json.dumps(cert, indent=2) + "\n"
    )
    print(f"\nSTEP 3 GATE: {cert['STEP3_GATE']}")
    print(f"Certificate: {outroot/'step3_certificate.json'}")
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
