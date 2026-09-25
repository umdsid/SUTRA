from __future__ import annotations
import argparse,json,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd

from strata_native_mechanics.patches import partition_core_cells,add_halo
from strata_native_mechanics.corrections import CorrectionConfig,corrected_objects
from strata_native_mechanics.solver import SolverConfig,build_patch_system
from strata_native_mechanics.observability import ObservabilityConfig,patch_observability

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def load_base(project,s):
    c=project/"results"/"native_mechanics_cache"/s
    E=pd.read_parquet(c/"interfaces.parquet")
    J=pd.read_json(c/"junctions.jsonl",lines=True)
    C=pd.read_parquet(c/"cells.parquet")
    if "incident_interfaces" in J.columns:
        J["incident_interfaces"]=J["incident_interfaces"].apply(
            lambda x:list(x) if isinstance(x,(list,tuple,np.ndarray)) else [])
    P=pd.read_parquet(
        project/"results"/"corrections"/"v064"/s/"mechanical_boundary_persistence.parquet"
    )
    Ep,Jp,_=corrected_objects(E,J,P,CorrectionConfig(),"persistent_boundaries")
    jp=project/"results"/"corrections"/"v065"/s/"junctions_persistent_plus_recovered.jsonl"
    if jp.exists():
        Juse=pd.read_json(jp,lines=True)
        if "incident_interfaces" in Juse.columns:
            Juse["incident_interfaces"]=Juse["incident_interfaces"].apply(
                lambda x:list(x) if isinstance(x,(list,tuple,np.ndarray)) else [])
    else:
        Juse=Jp
    return E,Ep,Juse,C

def one_patch(project_s,s,pid,cells,dense_cap,nullity_cap,tol):
    os.environ.update(
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1"
    )
    project=Path(project_s)
    Eorig,E,J,C=load_base(project,s)
    A,b,meta=build_patch_system(cells,E,J,C,SolverConfig())
    cfg=ObservabilityConfig(
        dense_nvar_cap=dense_cap,
        dense_nullity_cap=nullity_cap,
        numerical_rel_tol=tol
    )
    sm,vdf,cdf=patch_observability(A,meta,meta["E"],cfg)
    sm.update({
        "sample":s,"patch_id":int(pid),"n_patch_cells":int(len(cells)),
        "n_rows":int(A.shape[0]),"n_variables":int(A.shape[1]),
        "structural_nullity":int(max(A.shape[1]-sm["matching_rank"],0)),
    })
    return s,pid,sm,vdf,cdf

def aggregate_sample(project,s,records,var_tables,contrast_tables):
    out=project/"results"/"corrections"/"v067"/s
    out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(records).sort_values("patch_id")
    df.to_parquet(out/"patch_observability.parquet",index=False)

    vv=[]
    for pid,v in var_tables:
        if len(v):
            x=v.copy();x.insert(0,"patch_id",pid);vv.append(x)
    if vv:
        pd.concat(vv,ignore_index=True).to_parquet(
            out/"variable_observability.parquet",index=False
        )
    cc=[]
    for pid,c in contrast_tables:
        if len(c):
            x=c.copy();x.insert(0,"patch_id",pid);cc.append(x)
    if cc:
        pd.concat(cc,ignore_index=True).to_parquet(
            out/"pressure_contrast_observability.parquet",index=False
        )

    def wmean(col,weightcol):
        q=df[col].notna() & (df[weightcol]>0)
        if not q.any(): return None
        return float(np.average(df.loc[q,col],weights=df.loc[q,weightcol]))

    summary={
        "sample":s,
        "n_patches":int(len(df)),
        "fully_structurally_identifiable_patch_fraction":
            float(np.mean(df.structural_nullity==0)),
        "median_structural_nullity":float(np.median(df.structural_nullity)),
        "q90_structural_nullity":float(np.quantile(df.structural_nullity,.9)),
        "structural_observable_tension_fraction":
            wmean("tension_structural_observable_fraction","tension_n"),
        "structural_observable_cell_pressure_fraction":
            wmean("cell_pressure_structural_observable_fraction","cell_pressure_n"),
        "structural_observable_boundary_pressure_fraction":
            wmean("boundary_pressure_structural_observable_fraction","boundary_pressure_n"),
        "numerically_checked_patch_fraction":
            float(np.mean(df.status=="PASS")),
        "numerical_observable_tension_fraction":
            wmean("tension_numerical_observable_fraction","tension_n"),
        "numerical_observable_cell_pressure_fraction":
            wmean("cell_pressure_numerical_observable_fraction","cell_pressure_n"),
        "numerical_observable_boundary_pressure_fraction":
            wmean("boundary_pressure_numerical_observable_fraction","boundary_pressure_n"),
        "numerical_observable_pressure_contrast_fraction":
            wmean("pressure_contrast_numerical_observable_fraction","pressure_contrast_n"),
        "unresolved_tail_patch_fraction_nullity_gt_20":
            float(np.mean(df.structural_nullity>20)),
        "unresolved_tail_patch_fraction_nullity_gt_100":
            float(np.mean(df.structural_nullity>100)),
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--patch-size",type=int,default=300)
    ap.add_argument("--halo",type=int,default=1)
    ap.add_argument("--dense-nvar-cap",type=int,default=450)
    ap.add_argument("--dense-nullity-cap",type=int,default=40)
    ap.add_argument("--numerical-rel-tol",type=float,default=1e-9)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    tasks=[];counts={}
    for s in SAMPLES:
        Eorig,E,J,C=load_base(project,s)
        cores=partition_core_cells(Eorig,patch_size=a.patch_size)
        patches=[add_halo(c,Eorig,hops=a.halo) for c in cores]
        counts[s]=len(patches)
        tasks.extend((s,pid,cells) for pid,cells in enumerate(patches))
        print(f"{s}: {len(patches)} patches",flush=True)

    print(f"\nStructural observability + bounded numerical verification ({a.workers} workers)",flush=True)
    rec={s:[] for s in SAMPLES};vars_={s:[] for s in SAMPLES};cons={s:[] for s in SAMPLES}
    done={s:0 for s in SAMPLES}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs={
            ex.submit(
                one_patch,str(project),s,pid,cells,
                a.dense_nvar_cap,a.dense_nullity_cap,a.numerical_rel_tol
            ):(s,pid)
            for s,pid,cells in tasks
        }
        for f in as_completed(futs):
            s,pid,sm,v,c=f.result()
            rec[s].append(sm);vars_[s].append((pid,v));cons[s].append((pid,c))
            done[s]+=1
            if done[s]==1 or done[s]%50==0 or done[s]==counts[s]:
                print(
                    f"  [{s}] {done[s]}/{counts[s]} "
                    f"nullity={sm['structural_nullity']} "
                    f"tau_obs={100*sm['tension_structural_observable_fraction']:.1f}% "
                    f"p_obs={100*sm['cell_pressure_structural_observable_fraction']:.1f}% "
                    f"num={sm['status']}",
                    flush=True
                )

    root=project/"results"/"corrections"/"v067"
    reports=[]
    for s in SAMPLES:
        sm=aggregate_sample(project,s,rec[s],vars_[s],cons[s])
        reports.append(sm)
        print(
            f"\n[DONE] {s}: patch_ident="
            f"{100*sm['fully_structurally_identifiable_patch_fraction']:.1f}% "
            f"tau_struct_obs={100*sm['structural_observable_tension_fraction']:.1f}% "
            f"p_struct_obs={100*sm['structural_observable_cell_pressure_fraction']:.1f}% "
            f"dp_num_obs={sm['numerical_observable_pressure_contrast_fraction']}",
            flush=True
        )

    cert={
        "strata_version":"0.6.7",
        "stage":"variable-level mechanical observability",
        "mechanical_representation":"v0.6.4 persistent boundaries + accepted v0.6.5 recovered junctions",
        "biological_geometry_modified":False,
        "mechanics_values_frozen":False,
        "observability_policy":{
            "structural":"Dulmage-Mendelsohn underdetermined-side reachability",
            "numerical":"SVD nullspace for bounded-size/bounded-nullity patches",
            "dense_nvar_cap":a.dense_nvar_cap,
            "dense_nullity_cap":a.dense_nullity_cap,
            "numerical_rel_tol":a.numerical_rel_tol,
            "unresolved_tail_is_not_regularized":True,
        },
        "sample_reports":reports,
    }
    root.mkdir(parents=True,exist_ok=True)
    (root/"observability_certificate.json").write_text(json.dumps(cert,indent=2))
    print(f"\nCertificate: {root/'observability_certificate.json'}",flush=True)

if __name__=="__main__":
    main()
