from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from sutra.io.discovery import discover_samples
from sutra.io.boundaries import read_boundary_table, polygons_from_boundary_table
from sutra.topology.calibration import CalibrationConfig, calibrate_sample


def _parse_eps(s):
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def main():
    ap = argparse.ArgumentParser(description="STRATA Tranche 2.1 — Gate B calibration")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--results-root", default=None)
    ap.add_argument(
        "--epsilons",
        default="0,0.025,0.05,0.075,0.10,0.15,0.20,0.25,0.35,0.50"
    )
    ap.add_argument("--min-shared-length", type=float, default=0.25)
    args = ap.parse_args()

    project = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else project/"data"
    results = Path(args.results_root).resolve() if args.results_root else project/"results"/"tranche2_1_gateB"
    results.mkdir(parents=True, exist_ok=True)

    cfg = CalibrationConfig(
        epsilons=_parse_eps(args.epsilons),
        min_shared_length=args.min_shared_length,
    )

    samples = discover_samples(data_root)
    if not samples:
        raise SystemExit("No Xenium cell-boundary parquet files discovered.")

    print("STRATA 0.2.4 | Tranche 2.1 | Gate B calibration")
    print(f"Project: {project}")
    print(f"Samples: {len(samples)}")
    print(f"Epsilons: {cfg.epsilons}")

    reports = []
    for k, sample in enumerate(samples, 1):
        name = sample["sample"]
        print(f"[{k}/{len(samples)}] {name}")
        out = results/name
        out.mkdir(parents=True, exist_ok=True)

        bdf = read_boundary_table(sample["cell_boundaries"])
        polygons, pdiag = polygons_from_boundary_table(bdf)

        sweep, pairs, confidence, cert = calibrate_sample(polygons, cfg)
        cert["sample"] = name
        cert["cell_boundaries"] = str(sample["cell_boundaries"])

        sweep.to_csv(out/"epsilon_sweep.csv", index=False)
        pairs.to_parquet(out/"candidate_pair_geometry.parquet", index=False)
        confidence.to_parquet(out/"certified_interfaces.parquet", index=False)
        (out/"gateB_sample_certificate.json").write_text(json.dumps(cert, indent=2))

        print(
            f"    epsilon*={cert['chosen_epsilon']:.3f} "
            f"plateau={'YES' if cert['plateau_found'] else 'NO'} "
            f"edges={cert['n_certified_edges']:,} "
            f"mean_degree={cert['mean_degree_at_chosen']:.3f} "
            f"isolated={100*cert['fraction_isolated_at_chosen']:.2f}% "
            f"high_conf={100*cert['high_confidence_fraction']:.1f}%"
        )
        reports.append(cert)

    chosen = [r["chosen_epsilon"] for r in reports]
    plateau_all = all(r["plateau_found"] for r in reports)

    # Cross-specimen topology rule: choose the maximum independently calibrated
    # epsilon, so no specimen is forced to use a threshold below its stability
    # onset. Gate B closes only if every sample has a detected plateau and the
    # calibrated values agree within one sweep step-scale (<=0.10).
    eps_spread = max(chosen) - min(chosen) if chosen else float("inf")
    gate_pass = plateau_all and eps_spread <= 0.10
    frozen_eps = max(chosen) if gate_pass else None

    overall = {
        "strata_version": "0.2.4",
        "tranche": "2.1",
        "gate": "B",
        "contract": "data-calibrated measured-boundary topology; no mechanics",
        "sample_certificates": reports,
        "plateau_all_samples": plateau_all,
        "calibrated_epsilon_spread": eps_spread,
        "gateB_status": "PASS" if gate_pass else "FAIL",
        "frozen_contact_epsilon": frozen_eps,
        "rule_if_pass": (
            "An interface is admissible only when measured polygon boundaries "
            "have minimum gap <= frozen_contact_epsilon and boundary-strip "
            "support >= min_shared_length."
        ) if gate_pass else None,
    }
    (results/"gateB_certificate.json").write_text(json.dumps(overall, indent=2))

    print()
    print(f"Gate B: {overall['gateB_status']}")
    if gate_pass:
        print(f"Frozen contact epsilon: {frozen_eps:.3f}")
    else:
        print("No topology threshold frozen; mechanics remains blocked.")
    print(f"Certificate: {results/'gateB_certificate.json'}")


if __name__ == "__main__":
    main()
