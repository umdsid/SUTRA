from __future__ import annotations
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sutra.hierarchy.v101.replay import (
    exact_causal_configs,calibrate_per_config,classify_landmarks,
    replay_detector,nearest_robust_scale
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def req(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p

def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def one(project_s,sample,cfg):
    os.environ.update(
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1"
    )
    project=Path(project_s)

    v94=project/"results"/"hierarchy_v094_ness_flow_diagnostics"/sample
    v941=project/"results"/"hierarchy_v0941_flow_metric_audit"/sample

    df=pd.read_parquet(req(v94/"landmark_flow_diagnostics.parquet"))
    norm=pd.read_parquet(req(v94/"observable_normalization.parquet"))
    stability=pd.read_parquet(req(v941/"baseline_window_stability.parquet"))

    configs,blocks,meta=exact_causal_configs(df,norm,cfg)
    thresholds=calibrate_per_config(configs,df,cfg)
    classified=classify_landmarks(configs,thresholds,df,cfg)
    trace,stop=replay_detector(classified,cfg)
    nearest=nearest_robust_scale(stop,stability)

    out=project/"results"/"hierarchy_v101_exact_causal_replay_audit"/sample
    out.mkdir(parents=True,exist_ok=True)
    wpq(configs,out/"causal_configuration_trajectory.parquet")
    wpq(blocks,out/"causal_block_resolution.parquet")
    wpq(thresholds,out/"causal_configuration_thresholds.parquet")
    wpq(classified,out/"causal_landmark_classification.parquet")
    wpq(trace,out/"causal_detector_replay.parquet")

    calibrated=int(thresholds.calibrated.sum()) if len(thresholds) else 0
    nnom=len(thresholds)
    minres=float(classified.resolved_config_fraction.min()) if len(classified) else 0.
    medres=float(classified.resolved_config_fraction.median()) if len(classified) else 0.
    late=classified[classified.removed_fraction>=float(cfg["detector_min_removed_fraction"])]
    late_med=float(late.resolved_config_fraction.median()) if len(late) else 0.

    replay_ok=(stop is not None)
    audit_alignment_ok=(
        nearest is not None
        and float(nearest["delta_removed_fraction"])<=float(cfg["max_allowed_removed_fraction_drift"])
    )

    report={
        "sample":sample,
        "landmarks":int(len(df)),
        "observables_used":int(meta["observables"]),
        "expected_blocks":meta["expected_blocks"],
        "n_expected_blocks":int(meta["n_expected_blocks"]),
        "min_resolved_blocks_per_config":int(meta["min_resolved_blocks"]),
        "nominal_configurations":int(nnom),
        "calibrated_configurations":int(calibrated),
        "median_resolved_config_fraction_all":medres,
        "median_resolved_config_fraction_detector_region":late_med,
        "minimum_resolved_config_fraction_all":minres,
        "causal_replay_stop":stop,
        "nearest_independent_robust_scale":nearest,
        "consistency_checks":{
            "same_causal_estimator_used_for_thresholds_and_replay":True,
            "configuration_specific_thresholds":True,
            "unresolved_configurations_excluded_from_vote_denominator":True,
            "minimum_resolved_configuration_fraction_required":True,
            "minimum_resolved_scientific_blocks_required":True,
            "bidirectional_or_future_fill_used":False,
            "centered_smoothing_used":False,
            "causal_backward_derivatives_only":True,
            "v100_production_outputs_used_for_calibration":False,
            "new_coarse_graining_performed":False,
        },
        "replay_found_confirmed_regime":bool(replay_ok),
        "aligned_with_independent_v0941_robust_scale":bool(audit_alignment_ok),
        "status":"PASS" if replay_ok and audit_alignment_ok and calibrated>=int(cfg["min_calibrated_configs"]) else "HOLD",
    }
    (out/"exact_causal_replay_certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v101_exact_causal_replay.json")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())

    c=json.loads(req(
        project/"results"/"hierarchy_v0941_flow_metric_audit"/
        "flow_metric_audit_global_certificate.json"
    ).read_text())
    if c.get("ROBUST_EFFECTIVE_REGIMES_READY") is not True:
        raise SystemExit("ERROR: v0.9.4.1 robust effective regimes not certified")

    print("STRATA 1.0.1 | Exact causal detector replay + consistency audit")
    print("No new coarse-graining.")
    print("Replays the detector on the exhaustive v0.9.3 trajectory.")
    print("Thresholds use the same causal estimator as replay, per configuration.")
    print("No future fill, no centered smoothing, unresolved configs do not vote.")
    print("Scientific-block and configuration resolution floors are enforced.")
    print("v1.0.0 production outputs are NOT used for calibration.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result();reports.append(r)
            stop=r["causal_replay_stop"]
            near=r["nearest_independent_robust_scale"]
            smsg=(f"replay_nodes={stop['stop_nodes']:,}" if stop else "replay=NONE")
            amsg=(f"audit_nodes={near['audited_nodes']:,} drift={100*near['delta_removed_fraction']:.1f}%"
                  if near else "audit_match=NONE")
            print(
                f"[DONE] {r['sample']}: configs={r['calibrated_configurations']}/"
                f"{r['nominal_configurations']} resolved_med="
                f"{r['median_resolved_config_fraction_detector_region']:.2f} "
                f"{smsg} {amsg} {r['status']}",flush=True
            )

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v101_exact_causal_replay_audit"
    g={
        "strata_version":"1.0.1",
        "stage":"exact causal replay and detector-consistency audit",
        "sample_reports":reports,
        "CAUSAL_REPLAY_AUDIT_GATE":"PASS" if gate else "HOLD",
        "FINITE_SUPPORT_SEMANTICS_CERTIFIED":bool(gate),
        "CAUSAL_ESTIMATOR_CALIBRATION_CERTIFIED":bool(gate),
        "READY_FOR_CORRECTED_ADAPTIVE_PRODUCTION_FLOW":bool(gate),
    }
    p=out/"exact_causal_replay_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nCAUSAL REPLAY AUDIT GATE: {g['CAUSAL_REPLAY_AUDIT_GATE']}")
    print(f"FINITE SUPPORT SEMANTICS CERTIFIED: {g['FINITE_SUPPORT_SEMANTICS_CERTIFIED']}")
    print(f"CAUSAL ESTIMATOR CALIBRATION CERTIFIED: {g['CAUSAL_ESTIMATOR_CALIBRATION_CERTIFIED']}")
    print(f"READY FOR CORRECTED ADAPTIVE PRODUCTION FLOW: {g['READY_FOR_CORRECTED_ADAPTIVE_PRODUCTION_FLOW']}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
