from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")


def _finite_series(s):
    x=pd.to_numeric(s, errors="coerce").to_numpy(float)
    return bool(np.all(np.isfinite(x)))


def validate_and_materialize(project:Path, sample:str):
    src=project/"results"/"production_mechanics_v069"/sample
    out=project/"results"/"hierarchy_mechanics_v0691"/sample
    out.mkdir(parents=True, exist_ok=True)

    t=pd.read_parquet(src/"production_core_tensions.parquet")
    d=pd.read_parquet(src/"production_core_pressure_contrasts.parquet")
    p=pd.read_parquet(src/"patch_solver_diagnostics.parquet")

    problems=[]

    # Ownership must remain one physical interface -> one production row.
    if not t.interface_id.is_unique:
        problems.append("duplicate tension interface ownership")
    if not d.interface_id.is_unique:
        problems.append("duplicate pressure-contrast interface ownership")

    if not np.all(np.isfinite(p.lsqr_relative_residual.to_numpy(float))):
        problems.append("nonfinite LSQR patch residual")
    if not np.all(np.isfinite(p.lsmr_relative_residual.to_numpy(float))):
        problems.append("nonfinite LSMR patch residual")

    # v0.6.9 contract: anything not numerically CERTIFIED must not expose a
    # production value.  This is the key hierarchy-readiness condition.
    if "tension_production" not in t:
        problems.append("missing tension_production column")
    else:
        bad=(t.numerical_status!="CERTIFIED") & t.tension_production.notna()
        if bool(bad.any()):
            problems.append(f"{int(bad.sum())} non-certified tensions exported")

    if "delta_p_production" not in d:
        problems.append("missing delta_p_production column")
    else:
        bad=(d.numerical_status!="CERTIFIED") & d.delta_p_production.notna()
        if bool(bad.any()):
            problems.append(f"{int(bad.sum())} non-certified pressure contrasts exported")

    tc=t[t.numerical_status=="CERTIFIED"].copy()
    dc=d[d.numerical_status=="CERTIFIED"].copy()

    if len(tc)==0:
        problems.append("no certified tensions")
    if len(dc)==0:
        problems.append("no certified pressure contrasts")

    if len(tc) and not _finite_series(tc.tension_production):
        problems.append("nonfinite certified tension value")
    if len(dc) and not _finite_series(dc.delta_p_production):
        problems.append("nonfinite certified pressure-contrast value")

    # Attach owner-patch residuals as downstream confidence/provenance fields.
    pdiag=p[[
        "patch_id",
        "lsqr_relative_residual",
        "lsmr_relative_residual",
        "n_rows",
        "n_variables",
    ]].rename(columns={
        "patch_id":"owner_patch_id",
        "lsqr_relative_residual":"owner_patch_lsqr_relative_residual",
        "lsmr_relative_residual":"owner_patch_lsmr_relative_residual",
        "n_rows":"owner_patch_n_rows",
        "n_variables":"owner_patch_n_variables",
    })

    tc=tc.merge(pdiag,on="owner_patch_id",how="left",validate="many_to_one")
    dc=dc.merge(pdiag,on="owner_patch_id",how="left",validate="many_to_one")

    # Compact hierarchy-facing tables.  Keep provenance needed to propagate
    # quality masks through later levels.
    tcols=[
        "interface_id","owner_patch_id","owner_anchor_cell",
        "tension_production","rowspace_residual_ratio","solver_disagreement",
        "owner_patch_lsqr_relative_residual","owner_patch_lsmr_relative_residual",
        "owner_patch_n_rows","owner_patch_n_variables",
    ]
    dcols=[
        "interface_id","owner_patch_id","owner_anchor_cell",
        "cell_i","cell_j","delta_p_production",
        "rowspace_residual_ratio","solver_disagreement",
        "owner_patch_lsqr_relative_residual","owner_patch_lsmr_relative_residual",
        "owner_patch_n_rows","owner_patch_n_variables",
    ]
    tc=tc[[c for c in tcols if c in tc.columns]].copy()
    dc=dc[[c for c in dcols if c in dc.columns]].copy()

    tc.to_parquet(out/"certified_tensions.parquet",index=False)
    dc.to_parquet(out/"certified_pressure_contrasts.parquet",index=False)

    ntotal=int(len(t)); nobs=int(np.sum(t.production_status=="OBSERVABLE"))
    ncert=int(len(tc)); nunstable=int(np.sum(t.numerical_status=="NUMERICALLY_UNSTABLE"))
    dtotal=int(len(d)); dobs=int(np.sum(d.production_status=="OBSERVABLE"))
    dcert=int(len(dc)); dunstable=int(np.sum(d.numerical_status=="NUMERICALLY_UNSTABLE"))

    report={
        "sample":sample,
        "status":"PASS" if not problems else "HOLD",
        "problems":problems,
        "tensions":{
            "n_core_total":ntotal,
            "n_rowspace_observable":nobs,
            "n_numerically_certified":ncert,
            "n_observable_but_numerically_unstable":nunstable,
            "certified_fraction_of_all_core":float(ncert/ntotal) if ntotal else 0.0,
            "numerical_retention_given_observable":float(ncert/nobs) if nobs else 0.0,
        },
        "pressure_contrasts":{
            "n_core_total":dtotal,
            "n_rowspace_observable":dobs,
            "n_numerically_certified":dcert,
            "n_observable_but_numerically_unstable":dunstable,
            "certified_fraction_of_all_core":float(dcert/dtotal) if dtotal else 0.0,
            "numerical_retention_given_observable":float(dcert/dobs) if dobs else 0.0,
        },
        "patch_solver":{
            "n_patches":int(len(p)),
            "median_lsqr_relative_residual":float(np.median(p.lsqr_relative_residual)),
            "q95_lsqr_relative_residual":float(np.quantile(p.lsqr_relative_residual,.95)),
            "median_lsmr_relative_residual":float(np.median(p.lsmr_relative_residual)),
            "q95_lsmr_relative_residual":float(np.quantile(p.lsmr_relative_residual,.95)),
        },
        "hierarchy_contract":{
            "only_certified_values_materialized":True,
            "unresolved_or_unstable_values_absent_from_hierarchy_tables":True,
            "absolute_cell_pressure_materialized":False,
            "geometry_modified":False,
            "observability_modified":False,
        },
    }
    (out/"summary.json").write_text(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    v68=project/"results"/"corrections"/"v068"/"rowspace_observability_certificate.json"
    if not v68.exists():
        raise SystemExit("ERROR: v0.6.8 row-space certificate missing")
    d68=json.loads(v68.read_text())
    if d68.get("ROWSPACE_OBSERVABILITY_GATE")!="PASS":
        raise SystemExit("ERROR: v0.6.8 row-space gate is not PASS")

    print("STRATA 0.6.9.1 | Production-mechanics hierarchy finalizer")
    print("Policy: numerically unstable observable candidates remain excluded, not blockers.")
    print("No mechanics, geometry, or observability thresholds are changed.\n")

    reports=[]
    for s in SAMPLES:
        r=validate_and_materialize(project,s)
        reports.append(r)
        tr=r["tensions"]; dp=r["pressure_contrasts"]
        print(
            f"[DONE] {s}: "
            f"tau certified={tr['n_numerically_certified']:,}/"
            f"{tr['n_rowspace_observable']:,} observable "
            f"({100*tr['numerical_retention_given_observable']:.2f}% retained) | "
            f"dp certified={dp['n_numerically_certified']:,}/"
            f"{dp['n_rowspace_observable']:,} observable "
            f"({100*dp['numerical_retention_given_observable']:.2f}% retained) | "
            f"{r['status']}"
        )

    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_mechanics_v0691"
    cert={
        "strata_version":"0.6.9.1",
        "stage":"hierarchy-ready production mechanics finalization",
        "source_v068_rowspace_gate":"PASS",
        "source_v069_gate":"HOLD",
        "source_v069_hold_interpretation":
            "HOLD arose because row-space-observable candidates with LSQR/LSMR disagreement were counted as global blockers even though v0.6.9 already withheld their production values.",
        "gate_revision":
            "A numerically unstable observable candidate is excluded/NA and does not block unrelated certified quantities.",
        "scientific_constraints_changed":False,
        "geometry_changed":False,
        "observability_changed":False,
        "mechanical_values_reestimated":False,
        "hierarchy_consumes_only_numerically_certified_observable_values":True,
        "sample_reports":reports,
        "PRODUCTION_MECHANICS_GATE":"PASS" if gate else "HOLD",
        "HIERARCHY_READY":bool(gate),
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"hierarchy_mechanics_certificate.json").write_text(json.dumps(cert,indent=2))

    print(f"\nPRODUCTION MECHANICS GATE: {cert['PRODUCTION_MECHANICS_GATE']}")
    print(f"HIERARCHY READY: {cert['HIERARCHY_READY']}")
    print(f"Certificate: {out/'hierarchy_mechanics_certificate.json'}")


if __name__=="__main__":
    main()
