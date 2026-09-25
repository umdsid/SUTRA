
from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from sutra.hierarchy.v1113.replay import verify_hashes,exact_replay,strict_historical_nodes

def one(project,sample,cfg,outdir):
    sdir=outdir/sample; sdir.mkdir(parents=True,exist_ok=True)
    replay,meta=exact_replay(project,sample,cfg)
    replay.to_csv(sdir/"exact_frozen_terminal_replay.csv",index=False)
    (sdir/"exact_frozen_terminal_replay_meta.json").write_text(json.dumps(meta,indent=2))
    hist,sources=strict_historical_nodes(project,sample,cfg)
    (sdir/"strict_historical_landmarks.json").write_text(json.dumps({
        "sample":sample,"nodes":hist,"sources":sources
    },indent=2))
    cov=meta["coverage"]
    ready=all(v>=cfg["minimum_replay_coverage"] for v in cov.values())
    status="PASS" if ready and bool(hist) else "HOLD"
    rep={
        "sample":sample,
        "coverage":cov,
        "used_function":meta.get("used_function") or meta.get("history_representation_used") or "evaluate_once/evaluate_persistent",
        "historical_nodes":hist,
        "status":status
    }
    (sdir/"exact_frozen_terminal_replay_report.json").write_text(json.dumps(rep,indent=2))
    return rep

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=".")
    ap.add_argument("--config",default="configs/hierarchy_v1113_exact_frozen_terminal_replay.json")
    a=ap.parse_args()
    project=Path(a.project).resolve()
    cfg=json.loads((project/a.config).read_text())
    outdir=project/"results"/"hierarchy_v1113_exact_frozen_terminal_replay"
    outdir.mkdir(parents=True,exist_ok=True)

    ok,hash_rows=verify_hashes(project,cfg["required_hashes"])
    (outdir/"frozen_source_hash_audit.json").write_text(json.dumps(hash_rows,indent=2))
    print("STRATA 1.1.1.3 | Exact frozen terminal-rule replay")
    print("No new coarse-graining. No threshold changes. No nearest imputation.")
    print("Frozen v1.0.10 terminal rule + v1.1.0 production source hashes are verified first.")
    print("The saved v1.1.0 scientific evaluations are replayed through the frozen terminal-rule API.")
    print("Historical landmark parsing accepts explicit node arrays/columns only; landmark counts are excluded.")
    print(f"Frozen source hash audit: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("Frozen source hash mismatch; refusing replay.")

    reps=[]
    with ProcessPoolExecutor(max_workers=cfg["parallel_specimens"]) as ex:
        fs={ex.submit(one,project,s,cfg,outdir):s for s in cfg["samples"]}
        for f in as_completed(fs):
            r=f.result(); reps.append(r)
            c=r["coverage"]
            print(f"[DONE] {r['sample']}: function={r['used_function']} "
                  f"coverage(m/e/s)={c['mass_status']:.3f}/{c['expr_status']:.3f}/{c['spatial_status']:.3f} "
                  f"historical_nodes={r['historical_nodes']} {r['status']}")

    reps=sorted(reps,key=lambda x:x["sample"])
    gate="PASS" if all(r["status"]=="PASS" for r in reps) else "HOLD"
    cert={
        "stage":"exact frozen terminal-rule replay",
        "strata_version":"1.1.1.3",
        "EXACT_FROZEN_TERMINAL_REPLAY_GATE":gate,
        "FROZEN_HASHES_VERIFIED":True,
        "THRESHOLDS_MODIFIED":False,
        "NEW_MERGES_PERFORMED":False,
        "NEAREST_IMPUTATION_USED":False,
        "HISTORICAL_LANDMARK_COUNTS_EXCLUDED":True,
        "READY_FOR_FINAL_TERMINAL_LANDSCAPE_REPLAY":gate=="PASS",
        "sample_reports":reps
    }
    cp=outdir/"exact_frozen_terminal_replay_global_certificate.json"
    cp.write_text(json.dumps(cert,indent=2))
    print(f"\nEXACT FROZEN TERMINAL REPLAY GATE: {gate}")
    print("FROZEN HASHES VERIFIED: True")
    print("THRESHOLDS MODIFIED: False")
    print("NEW MERGES PERFORMED: False")
    print(f"READY FOR FINAL TERMINAL LANDSCAPE REPLAY: {gate=='PASS'}")
    print(f"Certificate: {cp}")

if __name__=="__main__":
    main()
