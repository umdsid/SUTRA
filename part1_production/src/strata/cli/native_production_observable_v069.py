from __future__ import annotations

import argparse, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from strata_native_mechanics.patches import partition_core_cells, add_halo
from strata_native_mechanics.corrections import CorrectionConfig, corrected_objects
from strata_native_mechanics.solver import SolverConfig, build_patch_system
from strata_native_mechanics.production_observable import (
    ProductionSolveConfig,
    solve_patch_pair,
    interface_owner_map,
    extract_patch_observables,
    core_owned_table,
    overlap_validation,
    add_solver_stability_status,
)

SAMPLES = ("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")


def load_representation(project: Path, sample: str):
    cache = project / "results" / "native_mechanics_cache" / sample
    E0 = pd.read_parquet(cache / "interfaces.parquet")
    J0 = pd.read_json(cache / "junctions.jsonl", lines=True)
    C = pd.read_parquet(cache / "cells.parquet")
    if "incident_interfaces" in J0.columns:
        J0["incident_interfaces"] = J0["incident_interfaces"].apply(
            lambda x: list(x) if isinstance(x, (list, tuple, np.ndarray)) else []
        )

    P = pd.read_parquet(
        project / "results" / "corrections" / "v064" / sample /
        "mechanical_boundary_persistence.parquet"
    )
    E, Jp, _ = corrected_objects(
        E0, J0, P, CorrectionConfig(), "persistent_boundaries"
    )

    jp = (
        project / "results" / "corrections" / "v065" / sample /
        "junctions_persistent_plus_recovered.jsonl"
    )
    if jp.exists():
        J = pd.read_json(jp, lines=True)
        if "incident_interfaces" in J.columns:
            J["incident_interfaces"] = J["incident_interfaces"].apply(
                lambda x: list(x) if isinstance(x, (list, tuple, np.ndarray)) else []
            )
    else:
        J = Jp

    v68 = project / "results" / "corrections" / "v068" / sample
    V = pd.read_parquet(v68 / "variable_rowspace_observability.parquet")
    D = pd.read_parquet(v68 / "pressure_contrast_rowspace_observability.parquet")
    return E0, E, J, C, V, D


def one_patch(project_s, sample, patch_id, cells, cfgdict):
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    project = Path(project_s)
    E0, E, J, C, V, D = load_representation(project, sample)
    A, b, meta = build_patch_system(cells, E, J, C, SolverConfig())

    cfg = ProductionSolveConfig(**cfgdict)
    x1, x2, solver = solve_patch_pair(A, b, cfg)
    T, DP = extract_patch_observables(
        patch_id, x1, x2, meta, V, D, cfg
    )
    solver.update({
        "sample": sample,
        "patch_id": int(patch_id),
        "n_patch_cells": int(len(cells)),
        "n_rows": int(A.shape[0]),
        "n_variables": int(A.shape[1]),
        "n_observable_tensions": int(len(T)),
        "n_observable_pressure_contrasts": int(len(DP)),
    })
    return sample, patch_id, solver, T, DP


def source_v068_gate(project: Path):
    p = project / "results" / "corrections" / "v068" / "rowspace_observability_certificate.json"
    if not p.exists():
        return False, "MISSING"
    d = json.loads(p.read_text())
    return d.get("ROWSPACE_OBSERVABILITY_GATE") == "PASS", d.get("ROWSPACE_OBSERVABILITY_GATE")


def sample_summary(
    sample,
    patch_df,
    tprod,
    dprod,
    tover,
    dover,
    cfg: ProductionSolveConfig,
):
    def frac_cert(df):
        if len(df) == 0:
            return 0.0
        return float(np.mean(df.numerical_status == "CERTIFIED"))

    def exported_fraction(df):
        if len(df) == 0:
            return 0.0
        return float(np.mean(df.production_status == "OBSERVABLE"))

    return {
        "sample": sample,
        "n_patches": int(len(patch_df)),
        "all_patches_finite_solver_diagnostics": bool(
            np.all(np.isfinite(patch_df.lsqr_relative_residual)) and
            np.all(np.isfinite(patch_df.lsmr_relative_residual))
        ),
        "median_lsqr_relative_residual": float(np.median(patch_df.lsqr_relative_residual)),
        "q95_lsqr_relative_residual": float(np.quantile(patch_df.lsqr_relative_residual, 0.95)),
        "median_lsmr_relative_residual": float(np.median(patch_df.lsmr_relative_residual)),
        "q95_lsmr_relative_residual": float(np.quantile(patch_df.lsmr_relative_residual, 0.95)),
        "core_tension_observable_fraction": exported_fraction(tprod),
        "core_tension_certified_fraction": frac_cert(tprod),
        "core_pressure_contrast_observable_fraction": exported_fraction(dprod),
        "core_pressure_contrast_certified_fraction": frac_cert(dprod),
        "n_core_tensions": int(len(tprod)),
        "n_certified_core_tensions": int(np.sum(tprod.numerical_status == "CERTIFIED")) if len(tprod) else 0,
        "n_core_pressure_contrasts": int(len(dprod)),
        "n_certified_core_pressure_contrasts": int(np.sum(dprod.numerical_status == "CERTIFIED")) if len(dprod) else 0,
        "solver_agreement_tolerance": cfg.solver_agreement_tol,
        "n_unstable_observable_tensions": int(np.sum(tprod.numerical_status == "NUMERICALLY_UNSTABLE")) if len(tprod) else 0,
        "n_unstable_observable_pressure_contrasts": int(np.sum(dprod.numerical_status == "NUMERICALLY_UNSTABLE")) if len(dprod) else 0,
        "overlap_tension_n": int(len(tover)),
        "overlap_tension_median_max_relative_deviation": (
            float(np.median(tover.max_relative_deviation_from_median)) if len(tover) else None
        ),
        "overlap_pressure_contrast_n": int(len(dover)),
        "overlap_pressure_contrast_median_max_relative_deviation": (
            float(np.median(dover.max_relative_deviation_from_median)) if len(dover) else None
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patch-size", type=int, default=300)
    ap.add_argument("--halo", type=int, default=1)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--btol", type=float, default=1e-12)
    ap.add_argument("--solver-agreement-tol", type=float, default=1e-7)
    ap.add_argument("--maxiter-factor", type=int, default=12)
    a = ap.parse_args()

    project = Path(a.project_root).resolve()
    source_pass, source_status = source_v068_gate(project)
    if not source_pass:
        raise SystemExit(
            f"ERROR: v0.6.8 row-space gate must PASS before v0.6.9; got {source_status}"
        )

    cfg = ProductionSolveConfig(
        atol=a.atol,
        btol=a.btol,
        solver_agreement_tol=a.solver_agreement_tol,
        maxiter_factor=a.maxiter_factor,
    )
    cfgdict = {
        "atol": cfg.atol,
        "btol": cfg.btol,
        "solver_agreement_tol": cfg.solver_agreement_tol,
        "maxiter_factor": cfg.maxiter_factor,
    }

    print("STRATA 0.6.9 | Production observable mechanics")
    print("Geometry: frozen v0.6.4 persistent boundaries + v0.6.5 recovered junctions")
    print("Observability: frozen v0.6.8 numerical row-space certificates")
    print("Export: deterministic core-owned tensions and cell-cell pressure contrasts only")
    print("Unresolved quantities remain NA.\n", flush=True)

    tasks = []
    sample_data = {}
    for s in SAMPLES:
        E0, E, J, C, V, D = load_representation(project, s)
        cores = partition_core_cells(E0, patch_size=a.patch_size)
        patches = [add_halo(core, E0, hops=a.halo) for core in cores]
        owners, cell_owner = interface_owner_map(cores, E)
        sample_data[s] = {
            "E0": E0, "E": E, "J": J, "C": C, "V": V, "D": D,
            "cores": cores, "patches": patches, "owners": owners,
        }
        tasks.extend((s, pid, cells) for pid, cells in enumerate(patches))
        print(f"{s}: {len(patches)} patches, {len(owners)} owned interfaces", flush=True)

    prec = {s: [] for s in SAMPLES}
    trec = {s: [] for s in SAMPLES}
    drec = {s: [] for s in SAMPLES}
    done = {s: 0 for s in SAMPLES}
    counts = {s: len(sample_data[s]["patches"]) for s in SAMPLES}

    print(f"\nDual sparse least-squares production solve ({a.workers} workers)", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {
            ex.submit(one_patch, str(project), s, pid, cells, cfgdict): (s, pid)
            for s, pid, cells in tasks
        }
        for f in as_completed(futs):
            s, pid, sm, T, DP = f.result()
            prec[s].append(sm)
            if len(T):
                T = T.copy()
                T.insert(0, "sample", s)
                trec[s].append(T)
            if len(DP):
                DP = DP.copy()
                DP.insert(0, "sample", s)
                drec[s].append(DP)
            done[s] += 1
            if done[s] == 1 or done[s] % 50 == 0 or done[s] == counts[s]:
                print(
                    f"  [{s}] {done[s]}/{counts[s]} "
                    f"nvar={sm['n_variables']} "
                    f"tau={sm['n_observable_tensions']} "
                    f"dp={sm['n_observable_pressure_contrasts']} "
                    f"r={sm['lsqr_relative_residual']:.3e}",
                    flush=True,
                )

    outroot = project / "results" / "production_mechanics_v069"
    outroot.mkdir(parents=True, exist_ok=True)

    reports = []
    gate = True

    for s in SAMPLES:
        sd = outroot / s
        sd.mkdir(parents=True, exist_ok=True)

        P = pd.DataFrame(prec[s]).sort_values("patch_id")
        T = pd.concat(trec[s], ignore_index=True) if trec[s] else pd.DataFrame()
        DP = pd.concat(drec[s], ignore_index=True) if drec[s] else pd.DataFrame()
        owners = sample_data[s]["owners"]

        tprod = core_owned_table(T, owners, "tension")
        dprod = core_owned_table(DP, owners, "pressure_contrast")
        tprod = add_solver_stability_status(tprod, cfg)
        dprod = add_solver_stability_status(dprod, cfg)

        tover = overlap_validation(T, "tension")
        dover = overlap_validation(DP, "delta_p")

        P.to_parquet(sd / "patch_solver_diagnostics.parquet", index=False)
        T.to_parquet(sd / "all_patch_observable_tensions.parquet", index=False)
        DP.to_parquet(sd / "all_patch_observable_pressure_contrasts.parquet", index=False)
        tprod.to_parquet(sd / "production_core_tensions.parquet", index=False)
        dprod.to_parquet(sd / "production_core_pressure_contrasts.parquet", index=False)
        tover.to_parquet(sd / "overlap_tension_validation.parquet", index=False)
        dover.to_parquet(sd / "overlap_pressure_contrast_validation.parquet", index=False)

        sm = sample_summary(s, P, tprod, dprod, tover, dover, cfg)
        sample_pass = (
            sm["all_patches_finite_solver_diagnostics"] and
            sm["n_unstable_observable_tensions"] == 0 and
            sm["n_unstable_observable_pressure_contrasts"] == 0
        )
        sm["production_gate"] = "PASS" if sample_pass else "HOLD"
        gate &= sample_pass

        (sd / "summary.json").write_text(json.dumps(sm, indent=2))
        reports.append(sm)

        print(
            f"\n[DONE] {s}: "
            f"core_tau_obs={100*sm['core_tension_observable_fraction']:.1f}% "
            f"core_dp_obs={100*sm['core_pressure_contrast_observable_fraction']:.1f}% "
            f"unstable_tau={sm['n_unstable_observable_tensions']} "
            f"unstable_dp={sm['n_unstable_observable_pressure_contrasts']} "
            f"status={sm['production_gate']}",
            flush=True,
        )

    cert = {
        "strata_version": "0.6.9",
        "stage": "production observable mechanics",
        "source_rowspace_gate": source_status,
        "geometry_frozen": True,
        "mechanical_representation":
            "v0.6.4 persistent boundaries + accepted v0.6.5 recovered junctions",
        "observability_frozen": True,
        "observability_source": "v0.6.8 numerical row-space certificate",
        "core_ownership": {
            "cell_cell_interface": "core containing smaller incident cell ID",
            "cell_background_interface": "core containing incident tissue cell",
            "observability_not_used_to_choose_owner": True,
        },
        "solver": {
            "internal_methods": ["LSQR", "LSMR"],
            "atol": cfg.atol,
            "btol": cfg.btol,
            "solver_agreement_tolerance": cfg.solver_agreement_tol,
            "minimum_norm_solution_has_no_physical_interpretation": True,
            "only_rowspace_invariant_quantities_are_exported": True,
        },
        "production_outputs": [
            "core-owned certified interface tensions",
            "core-owned certified cell-cell pressure contrasts",
        ],
        "absolute_cell_pressure_exported": False,
        "unresolved_quantities": "NA",
        "overlap_validation_defines_observability": False,
        "sample_reports": reports,
        "PRODUCTION_MECHANICS_GATE": "PASS" if gate else "HOLD",
        "hierarchy_ready": bool(gate),
    }
    (outroot / "production_mechanics_certificate.json").write_text(
        json.dumps(cert, indent=2)
    )

    print(
        f"\nPRODUCTION MECHANICS GATE: {cert['PRODUCTION_MECHANICS_GATE']}",
        flush=True,
    )
    print(f"HIERARCHY READY: {cert['hierarchy_ready']}", flush=True)
    print(
        f"Certificate: {outroot/'production_mechanics_certificate.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
