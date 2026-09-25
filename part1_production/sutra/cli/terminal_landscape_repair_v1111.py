
from __future__ import annotations
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from strata_hierarchy.v1111.audit import analyze

def _one(project,sample,cfg,outdir):
    return analyze(project,sample,cfg,outdir)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=".")
    ap.add_argument("--config",default="configs/hierarchy_v1111_terminal_landscape_repair.json")
    a=ap.parse_args()
    project=Path(a.project).resolve()
    cfg=json.loads((project/a.config).read_text())
    outdir=project/"results"/"hierarchy_v1111_terminal_landscape_repair"
    outdir.mkdir(parents=True,exist_ok=True)

    print("STRATA 1.1.1.1 | Terminal-landscape support + basin repair")
    print("No new reduction. No merge decisions. No node-count target.")
    print("Reads the original v1.1.0 scientific_evaluations.parquet directly.")
    print("PASS/HOLD support fields are resolved by schema, not silently dropped.")
    print("Persistence basins require nearest enclosing local shoulders on both sides.")
    print("Previous frozen hierarchy landmarks are compared by exact node and log-scale distance.")
    print(f"Parallel specimen workers: {cfg['parallel_specimens']}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=cfg["parallel_specimens"]) as ex:
        fut={ex.submit(_one,project,s,cfg,outdir):s for s in cfg["samples"]}
        for f in as_completed(fut):
            r=f.result(); reps.append(r)
            cov=r["support_coverage"]
            print(f"[DONE] {r['sample']}: evals={r['evaluations']} blocks={r['speed_blocks']} "
                  f"speed_full={r['full_speed_support_after_eval0']:.3f} "
                  f"support(m/e/s)={cov['mass']:.3f}/{cov['expr']:.3f}/{cov['spatial']:.3f} "
                  f"basins={r['basins']} pareto={r['pareto_landmarks']} "
                  f"nodes={r['pareto_nodes']} {r['status']}")
    reps=sorted(reps,key=lambda x:x["sample"])
    gate="PASS" if all(r["status"]=="PASS" for r in reps) else "HOLD"
    cert={
        "stage":"terminal landscape support + basin repair",
        "strata_version":"1.1.1.1",
        "TERMINAL_LANDSCAPE_REPAIR_GATE":gate,
        "NEW_MERGES_PERFORMED":False,
        "NODE_COUNT_TARGET_USED":False,
        "MASS_IS_EXISTENCE_VETO":False,
        "SUPPORT_FIELDS_REQUIRED":True,
        "LOCAL_ENCLOSING_SHOULDERS_REQUIRED":True,
        "HISTORICAL_ALIGNMENT_EXPORTED":True,
        "SCIENTIFIC_LANDMARKS_READY":gate=="PASS",
        "sample_reports":reps
    }
    cp=outdir/"terminal_landscape_repair_global_certificate.json"
    cp.write_text(json.dumps(cert,indent=2))
    print(f"\nTERMINAL LANDSCAPE REPAIR GATE: {gate}")
    print("NEW MERGES PERFORMED: False")
    print("SUPPORT FIELDS REQUIRED: True")
    print("LOCAL ENCLOSING SHOULDERS REQUIRED: True")
    print(f"SCIENTIFIC LANDMARKS READY: {gate=='PASS'}")
    print(f"Certificate: {cp}")

if __name__=="__main__":
    main()
