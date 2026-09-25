from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from strata_hierarchy.v071.resource_intake import sha256_file
from strata_hierarchy.v076.pressure_field_finalize import (
    decompose_pressure_field,decomposition_summary,component_summary,array_sha256
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def require(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p

def writepq(df,path):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),path,compression="zstd")

def one(project,sample):
    l0=project/"results"/"hierarchy_level0_v070"/sample
    v72=project/"results"/"hierarchy_v072_short_pilot"/sample
    v761=project/"results"/"hierarchy_v0761_pressure_tail_audit"/sample

    cells=pd.read_parquet(require(l0/"cells.parquet"))
    edge_rel=pd.read_parquet(require(v72/"level0_candidate_relations.parquet"))

    cell,E,solver=decompose_pressure_field(edge_rel,len(cells))
    summary=decomposition_summary(E)
    comp=component_summary(E)

    old=json.loads(require(v761/"pressure_tail_audit.json").read_text())
    oldtail=old["tail_summary"]

    reproduce={
        "median_abs_error":float(abs(summary["residual_median_abs"]-float(oldtail["median_abs"]))),
        "q95_abs_error":float(abs(summary["residual_q95_abs"]-float(oldtail["q95_abs"]))),
        "q99_abs_error":float(abs(summary["residual_q99_abs"]-float(oldtail["q99_abs"]))),
        "q999_abs_error":float(abs(summary["residual_q999_abs"]-float(oldtail["q999_abs"]))),
        "max_abs_error":float(abs(summary["residual_max_abs"]-float(oldtail["max_abs"]))),
        "rms_abs_error":float(abs(summary["residual_rms"]-float(oldtail["rms"]))),
    }
    reproduce["matches_v0761"]=bool(
        reproduce["median_abs_error"]<=1e-8
        and reproduce["q95_abs_error"]<=1e-8
        and reproduce["q99_abs_error"]<=1e-8
        and reproduce["q999_abs_error"]<=1e-8
        and reproduce["max_abs_error"]<=max(1e-5,1e-12*abs(float(oldtail["max_abs"])))
        and reproduce["rms_abs_error"]<=max(1e-6,1e-10*abs(float(oldtail["rms"])))
    )

    out=project/"results"/"hierarchy_v0762_pressure_field"/sample
    out.mkdir(parents=True,exist_ok=True)
    writepq(cell,out/"pressure_potential_cells.parquet")
    writepq(E,out/"pressure_field_decomposition.parquet")
    writepq(comp,out/"pressure_field_components.parquet")
    writepq(
        E[["source_edge_row","cell_i_index","cell_j_index",
           "pressure_component","delta_p_potential"]],
        out/"transport_pressure_edges.parquet"
    )
    writepq(
        E[["source_edge_row","cell_i_index","cell_j_index",
           "pressure_component","delta_p_observed","delta_p_potential",
           "delta_p_residual","abs_delta_p_residual",
           "pressure_consistency_ratio","reconstruction_relative_error"]],
        out/"pressure_consistency_diagnostics.parquet"
    )

    report={
        "sample":sample,
        "solver":solver,
        "summary":summary,
        "v0761_reproduction":reproduce,
        "pressure_potential_sha256":array_sha256(cell.pressure_potential.to_numpy(np.float64)),
        "transport_contract":{
            "scalar_pressure_for_transport":"delta_p_potential",
            "nonpotential_residual_for_transport":False,
            "residual_preserved_as_diagnostic":True,
            "consistency_ratio_is_gate":False,
            "constraints_removed":False,
            "outliers_clipped":False,
        },
        "status":"PASS" if (
            summary["all_finite"]
            and summary["exact_additive_decomposition_float64"]
            and reproduce["matches_v0761"]
        ) else "HOLD",
    }
    (out/"pressure_field_certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    src=require(project/"results"/"hierarchy_v0761_pressure_tail_audit"/"pressure_tail_audit_certificate.json")
    d=json.loads(src.read_text())
    if d.get("READY_TO_DECIDE_PRESSURE_TAIL_POLICY") is not True:
        raise SystemExit("ERROR: v0.7.6.1 pressure-tail audit did not pass")

    print("STRATA 0.7.6.2.1 | Pressure-field finalization certificate fix")
    print("No pressure field is changed.")
    print("Additive identity is certified by scale-aware float64 backward error.\n")

    reports=[]
    for s in SAMPLES:
        r=one(project,s); reports.append(r)
        q=r["summary"]
        print(
            f"[DONE] {s}: constraints={q['n_constraints']:,} "
            f"recon_abs={q['reconstruction_max_abs_error']:.3e} "
            f"recon_rel={q['reconstruction_max_relative_error']:.3e} "
            f"eps_units={q['reconstruction_max_error_in_eps_units']:.2f} "
            f"res_q99={q['residual_q99_abs']:.3e} {r['status']}"
        )

    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v0762_pressure_field"
    cert={
        "strata_version":"0.7.6.2.1",
        "stage":"pressure-field finalization",
        "numerical_certificate_fix":{
            "old_rule":"fixed absolute reconstruction tolerance 1e-10",
            "new_rule":"scale-aware float64 backward error <= 64 machine eps",
            "data_changed":False,
            "pressure_solution_changed":False,
            "residual_field_changed":False,
            "transport_policy_changed":False,
        },
        "source_v0761_certificate_sha256":sha256_file(src),
        "mathematical_decomposition":{
            "observed":"delta_p",
            "potential":"p_i-p_j",
            "residual":"delta_p-(p_i-p_j)",
            "identity":"delta_p = potential + residual",
        },
        "transport_policy":{
            "uses_potential_component":True,
            "uses_nonpotential_residual_as_scalar_pressure":False,
            "residual_retained_for_consistency_and_cycle_analysis":True,
            "pressure_consistency_ratio_is_diagnostic_only":True,
            "no_constraints_removed":True,
            "no_outliers_clipped":True,
            "hierarchy_rerun":False,
        },
        "sample_reports":reports,
        "PRESSURE_FIELD_FINALIZATION_GATE":"PASS" if gate else "HOLD",
        "PRESSURE_FIELD_FROZEN":bool(gate),
        "READY_FOR_DISCRETE_TRANSPORT":bool(gate),
    }
    p=out/"pressure_field_global_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print(f"\nPRESSURE FIELD FINALIZATION GATE: {cert['PRESSURE_FIELD_FINALIZATION_GATE']}")
    print(f"PRESSURE FIELD FROZEN: {cert['PRESSURE_FIELD_FROZEN']}")
    print(f"READY FOR DISCRETE TRANSPORT: {cert['READY_FOR_DISCRETE_TRANSPORT']}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
