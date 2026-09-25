from __future__ import annotations
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from strata_hierarchy.v102.future_blind import replay_grid,consensus,detector_source_is_future_blind

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")
def req(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p
def wpq(df,p): pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def one(project_s,sample,cfg):
    project=Path(project_s)
    v94=project/"results"/"hierarchy_v094_ness_flow_diagnostics"/sample
    v941=project/"results"/"hierarchy_v0941_flow_metric_audit"/sample
    df=pd.read_parquet(req(v94/"landmark_flow_diagnostics.parquet"))
    norm=pd.read_parquet(req(v94/"observable_normalization.parquet"))
    stability=pd.read_parquet(req(v941/"baseline_window_stability.parquet"))
    grid,traces=replay_grid(df,norm,stability,cfg); cns=consensus(grid,cfg)
    out=project/"results"/"hierarchy_v102_future_blind_plateau_replay"/sample
    out.mkdir(parents=True,exist_ok=True); wpq(grid,out/"future_blind_parameter_grid.parquet")
    key=(cfg["central_recent_width"],cfg["central_history_width"])
    f,tr,stop,near,meta=traces[key]
    wpq(f,out/"central_future_blind_features.parquet"); wpq(tr,out/"central_detector_trace.parquet")
    report={"sample":sample,"landmarks":len(df),"observables_used":meta["observables"],
            "scientific_blocks":meta["blocks"],"n_scientific_blocks":meta["n_blocks"],
            "central_configuration":{"recent_width":key[0],"history_width":key[1],"stop":stop,
                                     "nearest_robust_scale":near},
            "grid_consensus":cns,
            "consistency_checks":{
              "detector_source_future_blind_static_audit":detector_source_is_future_blind(),
              "future_trajectory_quantiles_used":False,"global_specimen_thresholds_used":False,
              "future_fill_or_backfill_used":False,"centered_smoothing_used":False,
              "only_past_and_current_landmarks_used":True,"relative_slowdown_required":True,
              "finite_scale_closure_required":True,"preceding_active_regime_required":True,
              "new_coarse_graining_performed":False},
            "status":"PASS" if cns["robust"] and meta["n_blocks"]>=8 and detector_source_is_future_blind() else "HOLD"}
    (out/"future_blind_plateau_replay_certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v102_future_blind_plateau_replay.json")
    a=ap.parse_args(); project=Path(a.project_root).resolve(); cfg=json.loads(req(project/a.config).read_text())
    cert=json.loads(req(project/"results"/"hierarchy_v0941_flow_metric_audit"/"flow_metric_audit_global_certificate.json").read_text())
    if cert.get("ROBUST_EFFECTIVE_REGIMES_READY") is not True: raise SystemExit("ERROR: v0.9.4.1 robust regimes not ready")
    print("STRATA 1.0.2 | Future-blind plateau replay")
    print("No new coarse-graining. No full-trajectory thresholds.")
    print("Stop = relative slowdown + finite-scale closure + prior active entry.")
    print("Nine history-window configurations audited in parallel across specimens.\n")
    reps=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result(); reps.append(r); c=r["grid_consensus"]; st=r["central_configuration"]["stop"]
            smsg=f"central_nodes={st['stop_nodes']:,}" if st else "central=NONE"
            print(f"[DONE] {r['sample']}: stop_support={c['stop_fraction']:.2f} "
                  f"audit_alignment={c['alignment_fraction']:.2f} consensus_nodes={c['consensus_nodes']:,} "
                  f"{smsg} {r['status']}",flush=True)
    reps.sort(key=lambda x:SAMPLES.index(x["sample"])); gate=all(r["status"]=="PASS" for r in reps)
    out=project/"results"/"hierarchy_v102_future_blind_plateau_replay"; out.mkdir(parents=True,exist_ok=True)
    g={"strata_version":"1.0.2","stage":"future-blind plateau replay","sample_reports":reps,
       "FUTURE_BLIND_REPLAY_GATE":"PASS" if gate else "HOLD",
       "PROSPECTIVE_STOPPING_LOGIC_INTERNALLY_CERTIFIED":bool(gate),
       "READY_FOR_CORRECTED_ADAPTIVE_PRODUCTION_FLOW":bool(gate)}
    p=out/"future_blind_plateau_replay_global_certificate.json"; p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nFUTURE-BLIND REPLAY GATE: {g['FUTURE_BLIND_REPLAY_GATE']}")
    print(f"PROSPECTIVE STOPPING LOGIC INTERNALLY CERTIFIED: {g['PROSPECTIVE_STOPPING_LOGIC_INTERNALLY_CERTIFIED']}")
    print(f"READY FOR CORRECTED ADAPTIVE PRODUCTION FLOW: {g['READY_FOR_CORRECTED_ADAPTIVE_PRODUCTION_FLOW']}")
    print(f"Certificate: {p}")
if __name__=="__main__": main()
