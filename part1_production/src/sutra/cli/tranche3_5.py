from __future__ import annotations
import argparse, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

from sutra.mechanics.dispersion_regularization import (
    calibrate_dispersion, neff_fraction, top_mass
)
from sutra.mechanics.stress_v2 import derive_stress


def _run_sample(payload):
    project_s, rep = payload
    project = Path(project_s)
    name = rep["sample"]
    t0 = time.time()

    src = project/"results"/"tranche3_2_gateC"/name
    edges = pd.read_parquet(src/"interface_mechanics.parquet")
    junctions = pd.read_parquet(src/"junction_geometry.parquet")
    pressure = pd.read_parquet(src/"cell_pressure.parquet")
    jr = pd.read_parquet(src/"junction_residuals.parquet")

    base_tau = edges["tension_like"].to_numpy(float)
    base_p = pressure["pressure_like"].to_numpy(float)
    base_resid = jr["fitted_residual"].to_numpy(float)
    cell_ids = pressure["cell_id"].astype(str).tolist()

    sweep, chosen, sol, cache = calibrate_dispersion(
        edges, cell_ids, len(junctions), base_tau, base_p, base_resid
    )

    out = project/"results"/"tranche3_5_dispersion"/name
    out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out/"dispersion_lambda_sweep.csv", index=False)

    if chosen is None:
        report = {
            "sample": name,
            "status": "FAIL",
            "reason": "no stable regularized plateau",
            "runtime_seconds": float(time.time() - t0),
        }
        (out/"dispersion_summary.json").write_text(json.dumps(report, indent=2))
        return report

    reg_edges = edges.copy()
    reg_edges["tension_like"] = sol["tension"]

    reg_pressure = pressure.copy()
    reg_pressure["pressure_like"] = sol["pressure"]

    cells = pd.read_parquet(list((project/"data"/name).rglob("*cells.parquet"))[0])
    cells["cell_id"] = cells["cell_id"].astype(str)
    stress = derive_stress(cells, reg_edges, reg_pressure)

    base_neff = neff_fraction(base_tau)
    reg_neff = neff_fraction(sol["tension"])
    base_top1 = top_mass(base_tau, 0.01)
    reg_top1 = top_mass(sol["tension"], 0.01)

    chosen_row = sweep.loc[sweep.lambda_tau == chosen].iloc[0]
    q95x = float(chosen_row.q95_residual_ratio)
    rho_prev = float(chosen_row.spearman_vs_previous_regularized)

    report = {
        "sample": name,
        "chosen_lambda_tau": chosen,
        "baseline_neff_fraction": base_neff,
        "regularized_neff_fraction": reg_neff,
        "baseline_top1_mass": base_top1,
        "regularized_top1_mass": reg_top1,
        "spearman_vs_previous_regularized": rho_prev,
        "q95_residual_ratio": q95x,
        "stress_valid_fraction": float(stress.stress_valid.mean()),
        "concentration_improved": bool(
            reg_neff >= 3.0*base_neff and reg_top1 <= 0.8*base_top1
        ),
        "plateau_stable": bool(rho_prev >= 0.90),
        "residual_ok": bool(q95x <= 1.25),
        "runtime_seconds": float(time.time() - t0),
        "status": "PASS",
    }

    reg_edges.to_parquet(out/"interface_mechanics_dispersion.parquet", index=False)
    reg_pressure.to_parquet(out/"cell_pressure_dispersion.parquet", index=False)
    stress.to_parquet(out/"cell_stress_dispersion.parquet", index=False)
    (out/"dispersion_summary.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--sample-workers", type=int, default=3)
    args = ap.parse_args()

    project = Path(args.project_root).resolve()
    gatec = json.loads(
        (project/"results"/"tranche3_2_gateC"/"gateC_certificate.json").read_text()
    )
    if gatec.get("gateC_status") != "PASS":
        raise SystemExit("Gate C must PASS first.")

    reps = gatec["sample_reports"]
    workers = min(max(1, args.sample_workers), len(reps))

    outroot = project/"results"/"tranche3_5_dispersion"
    outroot.mkdir(parents=True, exist_ok=True)

    print("STRATA 0.4.1 | Tranche 3.5 | Dispersion-regularized mechanics")
    print(f"Running {len(reps)} specimens on {workers} workers.", flush=True)

    reports = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_run_sample, (str(project), rep)): rep["sample"]
            for rep in reps
        }

        for fut in as_completed(futs):
            name = futs[fut]
            try:
                report = fut.result()
            except Exception as e:
                report = {
                    "sample": name,
                    "status": "FAIL",
                    "reason": f"{type(e).__name__}: {e}",
                }
            reports.append(report)

            if report.get("chosen_lambda_tau") is not None:
                print(
                    f"[DONE] {name}: "
                    f"lambda={report['chosen_lambda_tau']:.1e} "
                    f"Neff={100*report['baseline_neff_fraction']:.2f}%"
                    f"->{100*report['regularized_neff_fraction']:.2f}% "
                    f"top1={100*report['baseline_top1_mass']:.1f}%"
                    f"->{100*report['regularized_top1_mass']:.1f}% "
                    f"rho_plateau={report['spearman_vs_previous_regularized']:.3f} "
                    f"q95x={report['q95_residual_ratio']:.3f} "
                    f"status={report['status']}",
                    flush=True,
                )
            else:
                print(
                    f"[DONE] {name}: FAIL — {report.get('reason','')}",
                    flush=True,
                )

    order = {rep["sample"]: i for i, rep in enumerate(reps)}
    reports.sort(key=lambda r: order.get(r["sample"], 999))

    overall = {
        "strata_version": "0.4.1",
        "tranche": "3.5",
        "contract": "confidence-weighted tension-dispersion regularization with regularized-plateau calibration; strain excluded",
        "sample_reports": reports,
        "tranche3_5_status": (
            "PASS" if all(r.get("status") == "PASS" for r in reports) else "FAIL"
        ),
    }

    (outroot/"tranche3_5_certificate.json").write_text(
        json.dumps(overall, indent=2)
    )

    print()
    print(f"Tranche 3.5: {overall['tranche3_5_status']}")
    print(f"Certificate: {outroot/'tranche3_5_certificate.json'}")


if __name__ == "__main__":
    main()
