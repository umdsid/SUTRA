from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
import numpy as np

from strata_native_mechanics.geometry_fast import FastGeometryConfig,build_or_load_cache
from strata_native_mechanics.patches import partition_core_cells,add_halo
from strata_native_mechanics.solver import SolverConfig,solve_patch
from strata_native_mechanics.reconcile import reconcile_patch_solutions

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def geometry_job(project_s,sample):
    project=Path(project_s)
    mask=project/"results"/"production_geometry_r5"/sample/"production_segmentation.tif"
    cache=project/"results"/"native_mechanics_cache"/sample
    t=time.time()
    E,B,bg,J,C,hit=build_or_load_cache(mask,cache,FastGeometryConfig())
    return {
        "sample":sample,"cache_hit":hit,"seconds":time.time()-t,
        "n_interfaces":len(E),"n_junctions":len(J),"n_cells":len(C),
        "n_internal_gaps":int((B.kind=="internal_gap").sum()) if len(B) else 0,
    }

def load_cache(project,sample):
    c=project/"results"/"native_mechanics_cache"/sample
    E=pd.read_parquet(c/"interfaces.parquet")
    B=pd.read_parquet(c/"background_components.parquet")
    J=pd.read_json(c/"junctions.jsonl",lines=True)
    C=pd.read_parquet(c/"cells.parquet")
    # json reader may return arrays/tuples inconsistently
    if "incident_interfaces" in J.columns:
        J["incident_interfaces"]=J["incident_interfaces"].apply(
            lambda x:list(x) if isinstance(x,(list,tuple,np.ndarray)) else []
        )
    return E,B,J,C

def patch_job(project_s,sample,patch_id,cells):
    # Force numerical kernels to one thread in patch workers. Parallelism lives
    # at the worker-pool level.
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["OPENBLAS_NUM_THREADS"]="1"
    os.environ["VECLIB_MAXIMUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    project=Path(project_s)
    E,B,J,C=load_cache(project,sample)
    t=time.time()
    r=solve_patch(cells,E,J,C,SolverConfig())
    return sample,patch_id,r,time.time()-t

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--patch-size",type=int,default=300)
    ap.add_argument("--halo",type=int,default=1)
    ap.add_argument("--geometry-workers",type=int,default=3)
    ap.add_argument("--patch-workers",type=int,default=max(2,min(8,(os.cpu_count() or 8)-2)))
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    print("Stage A | vectorized geometry + persistent cache",flush=True)
    greports=[]
    with ProcessPoolExecutor(max_workers=min(3,a.geometry_workers)) as ex:
        futs={ex.submit(geometry_job,str(project),s):s for s in SAMPLES}
        for f in as_completed(futs):
            r=f.result();greports.append(r)
            print(
                f"  [GEOM] {r['sample']}: "
                f"{'CACHE' if r['cache_hit'] else 'BUILT'} "
                f"cells={r['n_cells']:,} interfaces={r['n_interfaces']:,} "
                f"junctions={r['n_junctions']:,} gaps={r['n_internal_gaps']} "
                f"{r['seconds']:.1f}s",flush=True
            )

    print(f"\nStage B | global patch pool ({a.patch_workers} workers)",flush=True)
    sample_data={}
    tasks=[]
    for s in SAMPLES:
        E,B,J,C=load_cache(project,s)
        cores=partition_core_cells(E,patch_size=a.patch_size)
        patches=[add_halo(core,E,hops=a.halo) for core in cores]
        sample_data[s]={"E":E,"B":B,"J":J,"C":C,"patches":patches,"results":[None]*len(patches)}
        for pid,cells in enumerate(patches):
            tasks.append((s,pid,cells))
        print(f"  {s}: {len(patches)} patches",flush=True)

    done={s:0 for s in SAMPLES}
    started=time.time()
    with ProcessPoolExecutor(max_workers=a.patch_workers) as ex:
        futs={
            ex.submit(patch_job,str(project),s,pid,cells):(s,pid)
            for s,pid,cells in tasks
        }
        for f in as_completed(futs):
            s,pid,r,sec=f.result()
            sample_data[s]["results"][pid]=r
            done[s]+=1
            n=len(sample_data[s]["patches"])
            if done[s]==1 or done[s]%10==0 or done[s]==n:
                print(
                    f"  [PATCH] {s}: {done[s]}/{n} "
                    f"last={sec:.2f}s status={r.get('status')} "
                    f"rank={100*r.get('identifiability',{}).get('structural_rank_fraction',0):.1f}% "
                    f"q95={r.get('residual_q95')}",flush=True
                )

    print("\nStage C | reconcile + certificates",flush=True)
    summaries=[]
    for s in SAMPLES:
        d=sample_data[s]
        results=d["results"]
        out=project/"results"/"native_mechanics_v061"/s
        out.mkdir(parents=True,exist_ok=True)

        tdf,pdf,bdf=reconcile_patch_solutions(results)
        tdf.to_parquet(out/"tensions.parquet",index=False)
        pdf.to_parquet(out/"pressures.parquet",index=False)
        bdf.to_parquet(out/"boundary_pressures.parquet",index=False)

        rows=[]
        for pid,r in enumerate(results):
            ident=r.get("identifiability",{})
            rows.append({
                "patch_id":pid,
                "n_patch_cells":len(d["patches"][pid]),
                "status":r.get("status"),
                "residual_rms":r.get("residual_rms"),
                "residual_q95":r.get("residual_q95"),
                "structural_rank":ident.get("structural_rank"),
                "n_variables":ident.get("n_variables"),
                "structural_rank_fraction":ident.get("structural_rank_fraction"),
                "structurally_identifiable":ident.get("structurally_identifiable"),
                "condition_estimate":ident.get("condition_estimate"),
            })
        pd.DataFrame(rows).to_csv(out/"patch_audit.csv",index=False)

        pass_frac=float(np.mean([r.get("status")=="PASS" for r in results])) if results else 0.
        ident_frac=float(np.mean([r.get("identifiability",{}).get("structurally_identifiable",False) for r in results])) if results else 0.
        q95=float(np.nanmedian([r.get("residual_q95",np.nan) for r in results])) if results else None
        summary={
            "sample":s,"version":"0.6.1",
            "method":"STRATA-native boundary-aware inverse mechanics",
            "n_cells":len(d["C"]),"n_interfaces":len(d["E"]),"n_junctions":len(d["J"]),
            "n_internal_gaps":int((d["B"].kind=="internal_gap").sum()) if len(d["B"]) else 0,
            "n_patches":len(results),
            "patch_pass_fraction":pass_frac,
            "patch_structural_identifiability_fraction":ident_frac,
            "median_patch_residual_q95":q95,
            "status":"PASS" if pass_frac>=.90 else "HOLD",
        }
        (out/"summary.json").write_text(json.dumps(summary,indent=2))
        summaries.append(summary)
        print(
            f"  [DONE] {s}: patch_pass={100*pass_frac:.1f}% "
            f"ident={100*ident_frac:.1f}% median_q95={q95} "
            f"status={summary['status']}",flush=True
        )

    rootout=project/"results"/"native_mechanics_v061"
    cert={
        "strata_version":"0.6.1",
        "accelerated":True,
        "geometry_cache":str(project/"results"/"native_mechanics_cache"),
        "patch_workers":a.patch_workers,
        "sample_reports":summaries,
        "gate_status":"PASS" if all(x["status"]=="PASS" for x in summaries) else "HOLD",
        "wall_seconds":time.time()-started,
    }
    (rootout/"native_mechanics_certificate.json").write_text(json.dumps(cert,indent=2))
    print(f"\nNATIVE MECHANICS GATE: {cert['gate_status']}",flush=True)
    print(f"Certificate: {rootout/'native_mechanics_certificate.json'}",flush=True)

if __name__=="__main__":main()
