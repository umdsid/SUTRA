
from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from strata_hierarchy.v112.final_landscape import analyze

def one(project,sample,cfg,outdir):
    return analyze(project,sample,cfg,outdir)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=".")
    ap.add_argument("--config",default="configs/hierarchy_v112_final_supported_hierarchy_landscape.json")
    a=ap.parse_args()
    project=Path(a.project).resolve()
    cfg=json.loads((project/a.config).read_text())

    # Hard prerequisites
    c1=project/"results"/cfg["exact_replay_stage"]/"exact_frozen_terminal_replay_global_certificate.json"
    c2=project/"results"/cfg["repaired_landscape_stage"]/"terminal_landscape_repair_global_certificate.json"
    if not c1.exists() or not c2.exists():
        raise SystemExit("Required v1.1.1.1/v1.1.1.3 result certificates are missing.")
    j1=json.loads(c1.read_text())
    if j1.get("EXACT_FROZEN_TERMINAL_REPLAY_GATE")!="PASS":
        raise SystemExit("Exact frozen terminal replay is not certified PASS.")

    outdir=project/"results"/"hierarchy_v112_final_supported_hierarchy_landscape"
    outdir.mkdir(parents=True,exist_ok=True)

    print("STRATA 1.1.2 | Final supported hierarchy landscape")
    print("No new reduction. No threshold changes. No node-count target.")
    print("v1.1.1.1 local persistence-basin geometry is preserved exactly.")
    print("v1.1.1.3 frozen-rule mass/expression/spatial states are joined by exact evaluation/node keys.")
    print("Pareto objectives are unchanged: lifetime, depth, block coherence, expression support, spatial support.")
    print("Mass remains descriptive/supporting evidence and is not an existence veto or Pareto objective.")
    print("Historical comparison uses the strict specimen-isolated v1.1.1.3 node lists.")
    print(f"Parallel specimen workers: {cfg['parallel_specimens']}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=cfg["parallel_specimens"]) as ex:
        fs={ex.submit(one,project,s,cfg,outdir):s for s in cfg["samples"]}
        for f in as_completed(fs):
            r=f.result(); reps.append(r)
            c=r["evaluation_support_coverage"]
            print(
                f"[DONE] {r['sample']}: support={c['mass']:.3f}/{c['expr']:.3f}/{c['spatial']:.3f} "
                f"basins={r['repaired_basins']} final={r['final_supported_landmarks']} "
                f"nodes={r['final_supported_nodes']} exact_history={r['exact_historical_matches']} {r['status']}"
            )

    reps=sorted(reps,key=lambda x:x["sample"])
    gate="PASS" if all(r["status"]=="PASS" for r in reps) else "HOLD"
    cert={
        "stage":"final supported hierarchy landscape",
        "strata_version":"1.1.2",
        "FINAL_SUPPORTED_HIERARCHY_GATE":gate,
        "EXACT_FROZEN_RULE_SUPPORT_USED":True,
        "REPAIRED_LOCAL_BASIN_GEOMETRY_PRESERVED":True,
        "PARETO_OBJECTIVES_CHANGED":False,
        "MASS_IS_EXISTENCE_VETO":False,
        "MASS_IS_PARETO_OBJECTIVE":False,
        "THRESHOLDS_MODIFIED":False,
        "NODE_COUNT_TARGET_USED":False,
        "NEW_MERGES_PERFORMED":False,
        "NEAREST_IMPUTATION_USED":False,
        "READY_FOR_HIERARCHY_ATLAS":gate=="PASS",
        "READY_FOR_MANUSCRIPT_BIOLOGY":gate=="PASS",
        "sample_reports":reps
    }
    cp=outdir/"final_supported_hierarchy_global_certificate.json"
    cp.write_text(json.dumps(cert,indent=2))

    print(f"\nFINAL SUPPORTED HIERARCHY GATE: {gate}")
    print("EXACT FROZEN-RULE SUPPORT USED: True")
    print("PARETO OBJECTIVES CHANGED: False")
    print("THRESHOLDS MODIFIED: False")
    print("NEW MERGES PERFORMED: False")
    print(f"READY FOR HIERARCHY ATLAS: {gate=='PASS'}")
    print(f"READY FOR MANUSCRIPT BIOLOGY: {gate=='PASS'}")
    print(f"Certificate: {cp}")

if __name__=="__main__":
    main()
