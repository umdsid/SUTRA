from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from strata.io.discovery import discover_samples
from strata.io.boundaries import read_boundary_table, polygons_from_boundary_table
from strata.topology.interfaces import TopologyConfig, build_observed_interfaces
from strata.topology.diagnostics import topology_diagnostics, reciprocity_audit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--min-shared-length", type=float, default=0.25)
    ap.add_argument("--contact-tolerance", type=float, default=0.25)
    args = ap.parse_args()

    project = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else project/"data"
    results = Path(args.results_root).resolve() if args.results_root else project/"results"/"tranche2_topology"
    results.mkdir(parents=True, exist_ok=True)

    cfg = TopologyConfig(min_shared_length=args.min_shared_length,
                         contact_tolerance=args.contact_tolerance)
    samples = discover_samples(data_root)
    print(f"STRATA 0.2.0 | Tranche 2")
    print(f"Project: {project}")
    print(f"Data:    {data_root}")
    print(f"Results: {results}")
    print(f"Samples: {len(samples)}")
    if not samples:
        raise SystemExit("No Xenium cell-boundary parquet files discovered.")

    summaries = []
    for k,s in enumerate(samples,1):
        print(f"[{k}/{len(samples)}] {s['sample']}")
        out = results/s["sample"]
        out.mkdir(parents=True, exist_ok=True)
        bdf = read_boundary_table(s["cell_boundaries"])
        polygons, pdiag = polygons_from_boundary_table(bdf)
        edges, nodes = build_observed_interfaces(polygons, cfg)
        diag = topology_diagnostics(pdiag, edges, nodes)
        recip = reciprocity_audit(edges)
        diag["reciprocity"] = recip
        diag["sample"] = s["sample"]
        diag["cell_boundaries"] = str(s["cell_boundaries"])
        diag["config"] = {
            "min_shared_length": cfg.min_shared_length,
            "contact_tolerance": cfg.contact_tolerance,
        }

        pdiag.to_parquet(out/"polygon_diagnostics.parquet", index=False)
        edges.to_parquet(out/"observed_interfaces.parquet", index=False)
        nodes.to_parquet(out/"topology_nodes.parquet", index=False)
        (out/"topology_summary.json").write_text(json.dumps(diag, indent=2))
        summaries.append(diag)
        print(f"    polygons={diag['n_polygons']:,} invalid={diag['n_invalid_polygons']:,} "
              f"interfaces={diag['n_observed_interfaces']:,} isolated={diag['n_isolated_cells']:,}")

    overall = {
        "strata_version":"0.2.0",
        "tranche":2,
        "contract":"observed topology only; no mechanics",
        "samples":summaries,
        "overall_status":"PASS" if all(x["reciprocity"]["status"]=="PASS" for x in summaries) else "FAIL",
    }
    (results/"tranche2_summary.json").write_text(json.dumps(overall, indent=2))
    print(f"Tranche 2 overall: {overall['overall_status']}")
    print(f"Summary: {results/'tranche2_summary.json'}")

if __name__ == "__main__":
    main()
