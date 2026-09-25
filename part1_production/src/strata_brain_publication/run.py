
from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from .core import ensure, bootstrap_stability, null_model_stability, build_gene_trajectory, read_support_trajectory
from .panels import set_style,hierarchy_landmarks,mass_spectrum_representatives,mass_stat_trajectory,merger_panels,gene_trajectory_panels,cell_subsampling_panel,scheme_null_panel,cross_brain_mass

def one(project,sample,cfg,rdir):
    a=bootstrap_stability(project,sample,cfg,rdir)
    b=null_model_stability(project,sample,cfg,rdir)
    c=build_gene_trajectory(project,sample,cfg,rdir)
    d=read_support_trajectory(project,sample,cfg,rdir)
    return {"sample":sample,"bootstrap_rows":len(a),"null_rows":len(b),"gene_rows":len(c),"support_rows":len(d)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=".")
    ap.add_argument("--config",default="configs/brain_publication_complete_v110.json")
    a=ap.parse_args()
    project=Path(a.project).resolve();cfg=json.loads((project/a.config).read_text())
    rdir=ensure(project/cfg["outputs"]["results"]);pbase=ensure(project/cfg["outputs"]["panels"])
    set_style(cfg)
    print("STRATA Brain Publication Complete v1.1.0")
    print("Frozen-hierarchy post-processing only.")
    print("Building statistical, merger-biology, and stability/null-control figure sets.")
    print()
    reps=[]
    with ProcessPoolExecutor(max_workers=3) as ex:
        fs={ex.submit(one,project,s["id"],cfg,rdir):s["id"] for s in cfg["samples"]}
        for f in as_completed(fs):
            r=f.result();reps.append(r)
            print(f"[ANALYSIS] {r['sample']}: bootstrap={r['bootstrap_rows']} null={r['null_rows']} gene={r['gene_rows']} PASS")
    manifest=[]
    for s in cfg["samples"]:
        sample=s["id"];pdir=ensure(pbase/sample)
        manifest+=hierarchy_landmarks(project,sample,cfg,pdir)
        manifest+=mass_spectrum_representatives(project,sample,cfg,pdir)
        manifest+=mass_stat_trajectory(project,sample,cfg,pdir)
        manifest+=merger_panels(project,sample,cfg,pdir)
        manifest+=gene_trajectory_panels(project,sample,cfg,rdir,pdir)
        manifest+=cell_subsampling_panel(sample,cfg,rdir,pdir)
        manifest+=scheme_null_panel(sample,cfg,rdir,pdir)
    manifest+=cross_brain_mass(project,cfg,ensure(pbase/"cohort"))
    mp=pbase/"panel_manifest.json";mp.write_text(json.dumps(manifest,indent=2))
    cert={
        "version":"1.1.0","status":"PASS","samples":[s["id"] for s in cfg["samples"]],
        "atomic_panel_count":len(manifest),"panel_manifest":str(mp),
        "hierarchy_changed":False,"new_merges_performed":False,"thresholds_modified":False,
        "cross_tissue_comparison_performed":False,
        "stability_scope":{
            "cell_subsampling":"performed on exact frozen partitions",
            "alternative_reduction_nulls":"random coalescent + spatial Voronoi at matched node counts",
            "full_objective_weight_reruns":"NOT performed in this package"
        }
    }
    cp=rdir/"brain_publication_complete_certificate.json";cp.write_text(json.dumps(cert,indent=2))
    print()
    print("BRAIN PUBLICATION COMPLETE: PASS")
    print("ATOMIC PANELS:",len(manifest))
    print("Panel manifest:",mp)
    print("Certificate:",cp)

if __name__=="__main__":main()
