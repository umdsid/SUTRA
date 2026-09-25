from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from strata.constants import VERSION
from strata.io.discovery import discover_xenium_samples
from strata.io.xenium import read_10x_h5_metadata, read_cells
from strata.level0.audit import audit_level0, write_level0_tables
from strata.provenance.hashing import sha256_file


def _load_config(project_root: Path, config_path: str | None) -> dict:
    p = Path(config_path) if config_path else project_root / "configs" / "tranche1.json"
    if not p.is_absolute():
        p = project_root / p
    return json.loads(p.read_text())


def _resolve(project_root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def main(argv=None):
    ap = argparse.ArgumentParser(description="STRATA Tranche 1 — measured Level-0 ingestion and contract audit")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-root", default=None, help="Override configured STRATA data root")
    ap.add_argument("--results-root", default=None)
    args = ap.parse_args(argv)

    project = Path(args.project_root).expanduser().resolve()
    cfg = _load_config(project, args.config)
    data_root = _resolve(project, args.data_root or cfg["data_root"])
    results_root = _resolve(project, args.results_root or cfg["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)

    samples = discover_xenium_samples(data_root, require_cells=bool(cfg.get("require_cells_table", True)))
    print(f"STRATA {VERSION} | Tranche 1")
    print(f"Project: {project}")
    print(f"Data:    {data_root}")
    print(f"Results: {results_root}")
    print(f"Samples: {len(samples)}")

    summary_samples = []
    overall_pass = True
    for i, sample in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {sample.sample_id}")
        out = results_root / sample.sample_id
        out.mkdir(parents=True, exist_ok=True)

        matrix = read_10x_h5_metadata(sample.matrix, audit_counts=bool(cfg.get("audit_sparse_counts", True)))
        cells = read_cells(sample.cells)
        audit = audit_level0(sample.sample_id, matrix, cells, strict_alignment=bool(cfg.get("strict_barcode_alignment", True)))
        write_level0_tables(out, matrix, cells)

        audit_payload = audit.to_dict()
        (out / "level0_audit.json").write_text(json.dumps(audit_payload, indent=2) + "\n")

        assets = sample.to_dict()
        provenance = {
            "schema_version": "strata.provenance.v1",
            "sample_id": sample.sample_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "strata_version": VERSION,
            "assets": assets,
            "hashes_sha256": {},
        }
        if cfg.get("hash_inputs", True):
            for key in ("matrix", "cells", "cell_boundaries", "nucleus_boundaries", "transcripts"):
                path = getattr(sample, key)
                if path is not None and Path(path).is_file():
                    provenance["hashes_sha256"][key] = sha256_file(path)
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (out / "measured_asset_inventory.json").write_text(json.dumps(assets, indent=2) + "\n")

        passed = audit.status == "PASS"
        overall_pass = overall_pass and passed
        print(f"    cells={audit.n_cells_matrix:,} features={audit.n_features:,} nnz={audit.nnz:,} status={audit.status}")
        summary_samples.append(audit_payload)

    summary = {
        "schema_version": "strata.tranche1.summary.v1",
        "strata_version": VERSION,
        "project_root": str(project),
        "data_root": str(data_root),
        "results_root": str(results_root),
        "n_samples": len(samples),
        "overall_status": "PASS" if overall_pass else "FAIL",
        "samples": summary_samples,
    }
    summary_path = results_root / "tranche1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Tranche 1 overall: {summary['overall_status']}")
    print(f"Summary: {summary_path}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
