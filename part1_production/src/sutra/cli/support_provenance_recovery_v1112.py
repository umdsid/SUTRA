
from __future__ import annotations
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
from sutra.hierarchy.v1112.recovery import (
    recover_supports,frozen_rule_inventory,isolated_historical_landmarks
)

def one(project,sample,cfg,outdir):
    src,recovered,inventory,prov,cov=recover_supports(project,sample,cfg)
    hist=isolated_historical_landmarks(project,sample,cfg)
    sdir=outdir/sample; sdir.mkdir(parents=True,exist_ok=True)
    recovered.to_csv(sdir/"recovered_terminal_supports.csv",index=False)
    (sdir/"support_source_inventory.json").write_text(json.dumps(inventory,indent=2))
    (sdir/"support_provenance.json").write_text(json.dumps(prov,indent=2))
    (sdir/"isolated_historical_landmarks.json").write_text(json.dumps(hist,indent=2))

    repair=project/"results"/cfg["repair_stage"]/sample/"pareto_landmarks_repaired.csv"
    alignment=[]
    if repair.exists() and hist:
        cand=pd.read_csv(repair)
        old=hist[0]["nodes"]
        import math
        for n in pd.to_numeric(cand.get("center_nodes",pd.Series(dtype=float)),errors="coerce").dropna():
            if not old: continue
            o=min(old,key=lambda x:abs(math.log(float(n)/float(x))))
            alignment.append({
                "candidate_nodes":int(n),
                "historical_stage":hist[0]["stage"],
                "nearest_historical_nodes":int(o),
                "absolute_node_difference":int(abs(n-o)),
                "relative_node_difference":float(abs(n-o)/o),
                "absolute_log_scale_distance":float(abs(math.log(float(n)/float(o))))
            })
    pd.DataFrame(alignment).to_csv(sdir/"isolated_historical_alignment.csv",index=False)

    support_ready=all(cov[k]>=cfg["minimum_exact_join_coverage"] for k in cfg["required_supports"])
    status="PASS" if support_ready and bool(hist) else "HOLD"
    rep={
        "sample":sample,
        "production_evaluation_source":str(src),
        "support_coverage":cov,
        "support_sources_found":len(inventory),
        "support_provenance":prov,
        "historical_stage":hist[0]["stage"] if hist else None,
        "historical_nodes":hist[0]["nodes"] if hist else [],
        "historical_alignment_rows":len(alignment),
        "status":status
    }
    (sdir/"support_provenance_recovery_report.json").write_text(json.dumps(rep,indent=2))
    return rep

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=".")
    ap.add_argument("--config",default="configs/hierarchy_v1112_support_provenance_recovery.json")
    a=ap.parse_args()
    project=Path(a.project).resolve()
    cfg=json.loads((project/a.config).read_text())
    outdir=project/"results"/"hierarchy_v1112_support_provenance_recovery"
    outdir.mkdir(parents=True,exist_ok=True)

    frozen=frozen_rule_inventory(project)
    (outdir/"frozen_terminal_rule_inventory.json").write_text(json.dumps(frozen,indent=2))

    print("STRATA 1.1.1.2 | Terminal support provenance recovery + historical isolation")
    print("No new coarse-graining. No threshold changes. No nearest imputation.")
    print("Scanning the complete saved v1.1.0 result tree for actually persisted PASS/HOLD state.")
    print("Support records are joined only by exact evaluation/node keys.")
    print("Historical landmarks are specimen-isolated; cross-specimen leakage is forbidden.")
    print("If support state was never persisted, this stage HOLDS and reports the frozen rule sources")
    print("needed for an exact reconstruction patch; it does not invent substitute thresholds.")
    print(f"Parallel specimen workers: {cfg['parallel_specimens']}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=cfg["parallel_specimens"]) as ex:
        fs={ex.submit(one,project,s,cfg,outdir):s for s in cfg["samples"]}
        for f in as_completed(fs):
            r=f.result(); reps.append(r)
            c=r["support_coverage"]
            print(f"[DONE] {r['sample']}: support_sources={r['support_sources_found']} "
                  f"coverage(m/e/s)={c['mass']:.3f}/{c['expr']:.3f}/{c['spatial']:.3f} "
                  f"historical={r['historical_stage']} nodes={r['historical_nodes']} {r['status']}")

    reps=sorted(reps,key=lambda x:x["sample"])
    all_support=all(all(r["support_coverage"][k]>=cfg["minimum_exact_join_coverage"]
                        for k in cfg["required_supports"]) for r in reps)
    all_hist=all(bool(r["historical_stage"]) for r in reps)
    gate="PASS" if all_support and all_hist else "HOLD"

    cert={
        "stage":"support provenance recovery + historical isolation",
        "strata_version":"1.1.1.2",
        "SUPPORT_PROVENANCE_RECOVERY_GATE":gate,
        "SUPPORT_STATE_RECOVERED":all_support,
        "HISTORICAL_LANDMARKS_SPECIMEN_ISOLATED":all_hist,
        "THRESHOLDS_MODIFIED":False,
        "NEAREST_IMPUTATION_USED":False,
        "NEW_MERGES_PERFORMED":False,
        "READY_FOR_EXACT_TERMINAL_RECONSTRUCTION_IF_NEEDED":not all_support,
        "READY_FOR_REPAIRED_LANDSCAPE_REPLAY":gate=="PASS",
        "frozen_terminal_rule_inventory":frozen,
        "sample_reports":reps
    }
    cp=outdir/"support_provenance_recovery_global_certificate.json"
    cp.write_text(json.dumps(cert,indent=2))
    print(f"\nSUPPORT PROVENANCE RECOVERY GATE: {gate}")
    print(f"SUPPORT STATE RECOVERED: {all_support}")
    print(f"HISTORICAL LANDMARKS SPECIMEN-ISOLATED: {all_hist}")
    print("THRESHOLDS MODIFIED: False")
    print("NEW MERGES PERFORMED: False")
    print(f"READY FOR REPAIRED LANDSCAPE REPLAY: {gate=='PASS'}")
    print(f"Certificate: {cp}")

if __name__=="__main__":
    main()
