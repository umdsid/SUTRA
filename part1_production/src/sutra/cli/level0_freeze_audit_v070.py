from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from sutra.io.discovery import discover_xenium_samples
from sutra.io.xenium import read_10x_h5_metadata
from sutra.hierarchy.freeze_audit import (
    Level0FreezeError,
    sha256_file,
    assert_exact_float_match,
    validate_csr_npz,
    validate_level0_interface_references,
    validate_retention_has_no_mechanics,
    canonical_table_digest,
)

SAMPLES = ("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sample_assets(project: Path):
    return {
        a.sample_id: a
        for a in discover_xenium_samples(project / "data", require_cells=True)
    }


def audit_one(project_s: str, sample: str):
    project = Path(project_s)
    out = project / "results" / "hierarchy_level0_v070" / sample
    if not out.exists():
        raise Level0FreezeError(f"{sample}: Step-3 output directory missing")

    summary = json.loads(
        require(out / "materialization_summary.json").read_text()
    )
    if summary.get("status") != "PASS":
        raise Level0FreezeError(f"{sample}: Step-3 summary is not PASS")

    cells = pd.read_parquet(require(out / "cells.parquet"))
    interfaces = pd.read_parquet(require(out / "interfaces.parquet"))
    patches = pd.read_parquet(require(out / "patches.parquet"))
    features = pd.read_parquet(require(out / "features.parquet"))
    expr_manifest = json.loads(
        require(out / "expression_manifest.json").read_text()
    )

    assets = sample_assets(project)[sample]
    matrix = read_10x_h5_metadata(assets.matrix, audit_counts=False)

    problems = []

    # A. Measured-data retention.
    measured = {
        "cells_matrix": int(matrix.shape[1]),
        "cells_level0": int(len(cells)),
        "features_matrix": int(matrix.shape[0]),
        "features_level0": int(len(features)),
        "cell_count_exact": bool(len(cells) == matrix.shape[1]),
        "feature_count_exact": bool(len(features) == matrix.shape[0]),
        "matrix_hash_exact": bool(
            expr_manifest["source_matrix_sha256"] == sha256_file(assets.matrix)
        ),
    }
    if not all(
        measured[k]
        for k in ("cell_count_exact", "feature_count_exact", "matrix_hash_exact")
    ):
        problems.append("measured-data retention mismatch")

    expected_cell_index = np.arange(len(cells), dtype=np.int64)
    if not np.array_equal(
        cells.cell_index.to_numpy(np.int64), expected_cell_index
    ):
        problems.append("noncanonical cell_index ordering")
    if not np.array_equal(
        cells.matrix_column.to_numpy(np.int64), expected_cell_index
    ):
        problems.append("matrix column order differs from Level-0 cell order")

    expected_feature_index = np.arange(len(features), dtype=np.int64)
    if not np.array_equal(
        features.feature_index.to_numpy(np.int64), expected_feature_index
    ):
        problems.append("noncanonical feature_index ordering")

    # B. Geometry and graph integrity.
    try:
        interface_refs = validate_level0_interface_references(
            interfaces, len(cells)
        )
        retention = validate_retention_has_no_mechanics(cells, interfaces)
        graph = validate_csr_npz(
            out / "graph_csr.npz", len(cells), interfaces
        )
    except Level0FreezeError as exc:
        problems.append(str(exc))
        interface_refs = {}
        retention = {}
        graph = {}

    # Patch table must contain all core IDs referenced by owned interfaces.
    patch_ids = set(patches.patch_id.astype(int))
    iface_patch_ids = set(interfaces.owner_patch_id.astype(int))
    if not iface_patch_ids.issubset(patch_ids):
        problems.append(
            f"{len(iface_patch_ids - patch_ids)} interface patch IDs absent from patch table"
        )

    # C. Exact mechanics source match.
    mech = project / "results" / "hierarchy_mechanics_v0691" / sample
    T = pd.read_parquet(require(mech / "certified_tensions.parquet"))
    DP = pd.read_parquet(
        require(mech / "certified_pressure_contrasts.parquet")
    )

    try:
        tau = assert_exact_float_match(
            interfaces[interfaces.tension.notna()],
            T,
            "interface_id",
            "tension",
            "tension_production",
            name=f"{sample} tension",
        )
        dp = assert_exact_float_match(
            interfaces[interfaces.delta_pressure.notna()],
            DP,
            "interface_id",
            "delta_pressure",
            "delta_p_production",
            name=f"{sample} delta-p",
        )
    except Level0FreezeError as exc:
        problems.append(str(exc))
        tau = {}
        dp = {}

    # Also require no extra Level-0 mechanics values beyond certified source.
    if int(interfaces.tension.notna().sum()) != len(T):
        problems.append("Level-0 tension count differs from certified source")
    if int(interfaces.delta_pressure.notna().sum()) != len(DP):
        problems.append("Level-0 delta-p count differs from certified source")

    # D. Upstream provenance hash exactness.
    pgeom = require(
        project / "results" / "production_geometry_r5" / sample /
        "production_geometry_summary.json"
    )
    pv68 = require(
        project / "results" / "corrections" / "v068" /
        "rowspace_observability_certificate.json"
    )
    pm = require(
        project / "results" / "hierarchy_mechanics_v0691" /
        "hierarchy_mechanics_certificate.json"
    )
    prov = summary.get("provenance", {})
    provenance = {
        "geometry_hash_exact": (
            prov.get("geometry_certificate_sha256") == sha256_file(pgeom)
        ),
        "observability_hash_exact": (
            prov.get("observability_certificate_sha256") == sha256_file(pv68)
        ),
        "mechanics_hash_exact": (
            prov.get("mechanics_certificate_sha256") == sha256_file(pm)
        ),
    }
    if not all(provenance.values()):
        problems.append("upstream provenance hash mismatch")

    # E. Deterministic content digests for future freeze verification.
    digests = {
        "cells_content_sha256": canonical_table_digest(
            cells, ["cell_index"]
        ),
        "interfaces_content_sha256": canonical_table_digest(
            interfaces, ["interface_id"]
        ),
        "patches_content_sha256": canonical_table_digest(
            patches, ["patch_id"]
        ),
        "features_content_sha256": canonical_table_digest(
            features, ["feature_index"]
        ),
        "graph_file_sha256": sha256_file(out / "graph_csr.npz"),
        "expression_manifest_sha256": sha256_file(
            out / "expression_manifest.json"
        ),
    }

    report = {
        "sample": sample,
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "measured_data": measured,
        "interface_integrity": interface_refs,
        "retention_integrity": retention,
        "graph_integrity": graph,
        "tension_source_match": tau,
        "pressure_contrast_source_match": dp,
        "provenance": provenance,
        "content_digests": digests,
    }

    (out / "freeze_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    project = Path(a.project_root).resolve()

    step3 = require(
        project / "results" / "hierarchy_level0_v070" /
        "step3_certificate.json"
    )
    s3 = json.loads(step3.read_text())
    if s3.get("STEP3_GATE") != "PASS":
        raise SystemExit("ERROR: Step 3 gate is not PASS")

    mech = require(
        project / "results" / "hierarchy_mechanics_v0691" /
        "hierarchy_mechanics_certificate.json"
    )
    md = json.loads(mech.read_text())
    if not md.get("HIERARCHY_READY", False):
        raise SystemExit("ERROR: upstream mechanics is not hierarchy-ready")

    print("STRATA 0.7.0 | Step 4 | Independent Level-0 freeze audit")
    print("No rebuilding. No repair. No inference.")
    print(f"Parallel specimen auditors: {min(a.workers, len(SAMPLES))}\n")

    reports = []
    with ProcessPoolExecutor(max_workers=min(a.workers, len(SAMPLES))) as ex:
        futs = {
            ex.submit(audit_one, str(project), s): s
            for s in SAMPLES
        }
        for f in as_completed(futs):
            r = f.result()
            reports.append(r)
            md = r["measured_data"]
            ri = r.get("retention_integrity", {})
            gi = r.get("graph_integrity", {})
            print(
                f"[DONE] {r['sample']}: "
                f"cells={md['cells_level0']:,} "
                f"features={md['features_level0']:,} "
                f"retention={ri.get('n_retention_cells','NA')} "
                f"geom_only_ifaces={ri.get('n_geometry_only_interfaces','NA')} "
                f"cell-cell={gi.get('n_cell_cell_interfaces','NA')} "
                f"{r['status']}",
                flush=True,
            )
            if r["problems"]:
                for p in r["problems"]:
                    print(f"    - {p}", flush=True)

    reports.sort(key=lambda x: SAMPLES.index(x["sample"]))
    gate = all(r["status"] == "PASS" for r in reports)

    outroot = project / "results" / "hierarchy_level0_v070"
    certificate = {
        "strata_version": "0.7.0",
        "stage": "Level-0 freeze",
        "step3_gate": "PASS",
        "mechanics_hierarchy_ready": True,
        "mechanics_recomputed": False,
        "geometry_modified": False,
        "expression_modified": False,
        "hierarchy_aggregation_performed": False,
        "freeze_policy": {
            "all_measured_cells_retained": True,
            "all_measured_features_retained": True,
            "geometry_only_objects_retained": True,
            "uncertified_mechanics_absent": True,
            "certified_mechanics_exact_source_copy": True,
            "csr_exact_interface_consistency": True,
            "upstream_certificate_hashes_exact": True,
        },
        "sample_reports": reports,
        "LEVEL0_FREEZE": "PASS" if gate else "FAIL",
        "HIERARCHY_INPUT_READY": bool(gate),
    }
    p = outroot / "level0_freeze_certificate.json"
    p.write_text(json.dumps(certificate, indent=2) + "\n")

    print(f"\nLEVEL-0 FREEZE: {certificate['LEVEL0_FREEZE']}")
    print(
        f"HIERARCHY INPUT READY: {certificate['HIERARCHY_INPUT_READY']}"
    )
    print(f"Certificate: {p}")

    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
