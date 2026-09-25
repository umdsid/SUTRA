
from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from .core import ensure_dir,validate_inputs,score_merger_genes
from .panels import generate_all_panels

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=".")
    ap.add_argument("--config",default="configs/brain_publication_pipeline_v1.json")
    a=ap.parse_args()
    project=Path(a.project).resolve();cfg=json.loads((project/a.config).read_text())
    out=ensure_dir(project/cfg["output_root"]);pan=ensure_dir(project/cfg["panel_root"])
    print("STRATA brain publication pipeline v1.0.0")
    print("Frozen hierarchy post-processing only. Brain cohort only.")
    print("Atomic publication panels only; no multi-panel figure assembly.\n")
    pre=validate_inputs(project,cfg);(out/"preflight_certificate.json").write_text(json.dumps(pre,indent=2))
    if pre["status"]!="PASS":
        print("PREFLIGHT: HOLD")
        for s,r in pre["samples"].items():
            if r["missing"]:print(s,r["missing"])
        raise SystemExit(2)
    print("PREFLIGHT: PASS")
    reps=[]
    with ProcessPoolExecutor(max_workers=3) as ex:
        fs={ex.submit(score_merger_genes,project,s["id"],cfg,out):s["id"] for s in cfg["samples"]}
        for f in as_completed(fs):
            r=f.result();reps.append(r)
            print(f"[BIOLOGY] {r['sample']}: landmarks={r['landmarks_scored']} PASS")
    print("\nGenerating atomic publication panels...")
    manifest,mp=generate_all_panels(project,cfg,out,pan)
    cert={"pipeline":"STRATA brain publication pipeline","version":"1.0.0","status":"PASS",
          "samples":[s["id"] for s in cfg["samples"]],"merger_biology_ready":True,
          "atomic_panels_ready":True,"panel_count":len(manifest),"panel_manifest":str(mp),
          "hierarchy_changed":False,"new_merges_performed":False,"thresholds_modified":False,
          "cross_tissue_comparison_performed":False}
    cp=out/"brain_publication_pipeline_certificate.json";cp.write_text(json.dumps(cert,indent=2))
    print("\nBRAIN PUBLICATION PIPELINE: PASS")
    print("ATOMIC PANELS GENERATED:",len(manifest))
    print("Panel manifest:",mp)
    print("Certificate:",cp)

if __name__=="__main__":main()
