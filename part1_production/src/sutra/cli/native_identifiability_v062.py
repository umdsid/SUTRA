import argparse, json, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from strata_native_mechanics.patches import partition_core_cells, add_halo
from strata_native_mechanics.solver import SolverConfig, build_patch_system
from strata_native_mechanics.identifiability_v2 import (
    BoundaryReductionConfig, nullity_attribution, equation_counts, reduction_audit
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def load_cache(project,s):
    c=project/"results"/"native_mechanics_cache"/s
    E=pd.read_parquet(c/"interfaces.parquet")
    B=pd.read_parquet(c/"background_components.parquet")
    J=pd.read_json(c/"junctions.jsonl",lines=True)
    C=pd.read_parquet(c/"cells.parquet")
    if "incident_interfaces" in J.columns:
        J["incident_interfaces"]=J["incident_interfaces"].apply(
            lambda x:list(x) if isinstance(x,(list,tuple,np.ndarray)) else [])
    return E,B,J,C

def job(project_s,s,pid,cells,small,medium):
    os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                      VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1")
    project=Path(project_s); E,B,J,C=load_cache(project,s)
    A,b,meta=build_patch_system(cells,E,J,C,SolverConfig())
    base,attr=nullity_attribution(A,meta)
    eq=equation_counts(meta)
    red=reduction_audit(A,meta,B,BoundaryReductionConfig(small,medium))
    return {"sample":s,"patch_id":pid,"n_patch_cells":len(cells),
            **base,**eq,"attribution":attr,"reductions":red}

def flatten(r):
    d={k:v for k,v in r.items() if k not in ("attribution","reductions")}
    for c,q in r["attribution"].items():
        d[f"{c}_nvars"]=q["n_variables_in_class"]
        d[f"{c}_nullity_removed"]=q["nullity_removed_if_class_removed"]
    for q in r["reductions"]:
        m=q["mode"]
        d[f"{m}_nullity"]=q["structural_nullity"]
        d[f"{m}_rank_fraction"]=q["structural_rank_fraction"]
        d[f"{m}_boundary_nvars"]=q["n_boundary_variables_reduced"]
        d[f"{m}_nullity_reduction"]=q["nullity_reduction_vs_independent"]
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--patch-size",type=int,default=300)
    ap.add_argument("--halo",type=int,default=1)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--small-gap-pixels",type=int,default=64)
    ap.add_argument("--medium-gap-pixels",type=int,default=256)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    tasks=[]; counts={}
    for s in SAMPLES:
        E,B,J,C=load_cache(project,s)
        cores=partition_core_cells(E,patch_size=a.patch_size)
        patches=[add_halo(c,E,hops=a.halo) for c in cores]
        counts[s]=len(patches)
        tasks.extend((s,i,p) for i,p in enumerate(patches))
        print(f"{s}: {len(patches)} patches",flush=True)

    rs=[]; done={s:0 for s in SAMPLES}; t0=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(job,str(project),s,i,p,a.small_gap_pixels,a.medium_gap_pixels):(s,i)
              for s,i,p in tasks}
        for f in as_completed(futs):
            r=f.result(); rs.append(r); s=r["sample"]; done[s]+=1
            if done[s]==1 or done[s]%50==0 or done[s]==counts[s]:
                print(f"  [{s}] {done[s]}/{counts[s]} "
                      f"rank={100*r['structural_rank_fraction']:.1f}% "
                      f"nullity={r['structural_nullity']}",flush=True)

    out=project/"results"/"native_identifiability_v062"; out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    modes=("independent","exterior_shared","small_gap_shared","tiered_gap_shared")
    for s in SAMPLES:
        df=pd.DataFrame([flatten(r) for r in rs if r["sample"]==s]).sort_values("patch_id")
        sd=out/s; sd.mkdir(parents=True,exist_ok=True)
        df.to_csv(sd/"patch_identifiability.csv",index=False)
        ms={}
        for m in modes:
            ms[m]={
                "identifiable_patch_fraction":float(np.mean(df[f"{m}_nullity"]==0)),
                "median_rank_fraction":float(df[f"{m}_rank_fraction"].median()),
                "median_nullity":float(df[f"{m}_nullity"].median()),
                "median_nullity_reduction_vs_independent":
                    float(df[f"{m}_nullity_reduction"].median())
            }
        best=max(modes,key=lambda m:(ms[m]["identifiable_patch_fraction"],
                                    ms[m]["median_rank_fraction"],
                                    -modes.index(m)))
        summ={
            "sample":s,
            "n_patches":len(df),
            "baseline_identifiable_patch_fraction":
                float(np.mean(df["structural_nullity"]==0)),
            "baseline_median_rank_fraction":
                float(df["structural_rank_fraction"].median()),
            "boundary_model_audit":ms,
            "best_structural_candidate":best,
            "small_gap_pixels":a.small_gap_pixels,
            "medium_gap_pixels":a.medium_gap_pixels,
        }
        (sd/"summary.json").write_text(json.dumps(summ,indent=2))
        summaries.append(summ)
        print(f"[DONE] {s}: baseline_ident="
              f"{100*summ['baseline_identifiable_patch_fraction']:.1f}% "
              f"best={best} best_ident="
              f"{100*ms[best]['identifiable_patch_fraction']:.1f}%",flush=True)

    cert={"strata_version":"0.6.2",
          "stage":"mechanical identifiability decomposition",
          "diagnostic_only":True,
          "mechanical_values_frozen":False,
          "sample_reports":summaries,
          "wall_seconds":time.time()-t0}
    (out/"identifiability_certificate.json").write_text(json.dumps(cert,indent=2))
    print(f"Certificate: {out/'identifiability_certificate.json'}")

if __name__=="__main__":
    main()
