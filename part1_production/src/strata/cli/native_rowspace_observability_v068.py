from __future__ import annotations

import argparse,json,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd

from strata_native_mechanics.patches import partition_core_cells,add_halo
from strata_native_mechanics.corrections import CorrectionConfig,corrected_objects
from strata_native_mechanics.solver import SolverConfig,build_patch_system
from strata_native_mechanics.rowspace_observability import (
    RowspaceConfig,patch_rowspace_observability,regression_confusion
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")


def load_base(project,s):
    c=project/"results"/"native_mechanics_cache"/s
    E=pd.read_parquet(c/"interfaces.parquet")
    J=pd.read_json(c/"junctions.jsonl",lines=True)
    C=pd.read_parquet(c/"cells.parquet")
    if "incident_interfaces" in J.columns:
        J["incident_interfaces"]=J["incident_interfaces"].apply(
            lambda x:list(x) if isinstance(x,(list,tuple,np.ndarray)) else []
        )

    P=pd.read_parquet(
        project/"results"/"corrections"/"v064"/s/
        "mechanical_boundary_persistence.parquet"
    )
    Ep,Jp,_=corrected_objects(E,J,P,CorrectionConfig(),"persistent_boundaries")

    jp=(project/"results"/"corrections"/"v065"/s/
        "junctions_persistent_plus_recovered.jsonl")
    if jp.exists():
        Juse=pd.read_json(jp,lines=True)
        if "incident_interfaces" in Juse.columns:
            Juse["incident_interfaces"]=Juse["incident_interfaces"].apply(
                lambda x:list(x) if isinstance(x,(list,tuple,np.ndarray)) else []
            )
    else:
        Juse=Jp
    return E,Ep,Juse,C


def one_patch(project_s,s,pid,cells,rank_rcond,obs_tol,marginal_tol):
    os.environ.update(
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1"
    )
    project=Path(project_s)
    Eorig,E,J,C=load_base(project,s)
    A,b,meta=build_patch_system(cells,E,J,C,SolverConfig())
    cfg=RowspaceConfig(
        rank_rcond=rank_rcond,
        observable_tol=obs_tol,
        marginal_tol=marginal_tol,
        row_normalize=True
    )
    sm,V,D=patch_rowspace_observability(A,meta,meta["E"],cfg)
    sm.update({
        "sample":s,"patch_id":int(pid),"n_patch_cells":int(len(cells)),
        "n_rows":int(A.shape[0]),"n_variables":int(A.shape[1]),
    })
    return s,pid,sm,V,D


def aggregate_global_object_mask(df,key_cols,object_type):
    """
    Conservative evidence aggregation across overlapping patches.

    A quantity is ROBUST_OBSERVABLE when at least two patch occurrences certify
    it observable. LOCAL_OBSERVABLE means exactly one certification. A failure
    elsewhere does not revoke a certification because patch truncation can only
    remove equations. We retain counts so downstream reconciliation can impose
    stricter rules if desired.
    """
    rows=[]
    for key,g in df.groupby(key_cols,dropna=False):
        if not isinstance(key,tuple):
            key=(key,)
        sts=g.rowspace_status.astype(str)
        nobs=int(np.sum(sts=="OBSERVABLE"))
        nmarg=int(np.sum(sts=="NUMERICALLY_MARGINAL"))
        nun=int(np.sum(sts=="UNRESOLVED"))
        if nobs>=2:
            status="ROBUST_OBSERVABLE"
        elif nobs==1:
            status="LOCAL_OBSERVABLE"
        elif nmarg>0:
            status="NUMERICALLY_MARGINAL"
        else:
            status="UNRESOLVED"
        rec={k:v for k,v in zip(key_cols,key)}
        rec.update({
            "object_type":object_type,
            "production_candidate_status":status,
            "n_patch_occurrences":int(len(g)),
            "n_observable_certifications":nobs,
            "n_marginal_occurrences":nmarg,
            "n_unresolved_occurrences":nun,
            "best_rowspace_residual_ratio":float(g.rowspace_residual_ratio.min()),
            "median_rowspace_residual_ratio":float(g.rowspace_residual_ratio.median()),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def regression_against_v067(project,s,V,D):
    oldroot=project/"results"/"corrections"/"v067"/s
    report={"sample":s}

    ov=oldroot/"variable_observability.parquet"
    if ov.exists():
        old=pd.read_parquet(ov)
        old=old[old.numerically_observable.notna()].copy()
        if len(old):
            new=V.merge(
                old[["patch_id","variable_class","object_id","numerically_observable"]],
                on=["patch_id","variable_class","object_id"],how="inner"
            )
            report["variables"]=regression_confusion(
                new.numerically_observable.astype(bool).to_numpy(),
                new.rowspace_status.to_numpy()
            )
        else:
            report["variables"]={"n":0}
    else:
        report["variables"]={"status":"MISSING_V067"}

    oc=oldroot/"pressure_contrast_observability.parquet"
    if oc.exists():
        old=pd.read_parquet(oc)
        old=old[old.numerically_observable.notna()].copy()
        if len(old):
            new=D.merge(
                old[["patch_id","interface_id","numerically_observable"]],
                on=["patch_id","interface_id"],how="inner"
            )
            report["pressure_contrasts"]=regression_confusion(
                new.numerically_observable.astype(bool).to_numpy(),
                new.rowspace_status.to_numpy()
            )
        else:
            report["pressure_contrasts"]={"n":0}
    else:
        report["pressure_contrasts"]={"status":"MISSING_V067"}
    return report


def weighted_fraction(df,col,ncol):
    q=df[col].notna() & (df[ncol]>0)
    if not q.any():
        return None
    return float(np.average(df.loc[q,col],weights=df.loc[q,ncol]))


def summarize_sample(patch,V,D):
    return {
        "n_patches":int(len(patch)),
        "numerically_checked_patch_fraction":1.0,
        "median_numerical_nullity":float(np.median(patch.numerical_nullity)),
        "q90_numerical_nullity":float(np.quantile(patch.numerical_nullity,.9)),
        "tension_observable_fraction":
            weighted_fraction(patch,"tension_observable_fraction","tension_n"),
        "cell_pressure_observable_fraction":
            weighted_fraction(patch,"cell_pressure_observable_fraction","cell_pressure_n"),
        "boundary_pressure_observable_fraction":
            weighted_fraction(patch,"boundary_pressure_observable_fraction","boundary_pressure_n"),
        "pressure_contrast_observable_fraction":
            weighted_fraction(patch,"pressure_contrast_observable_fraction","pressure_contrast_n"),
        "tension_marginal_fraction":
            weighted_fraction(patch,"tension_marginal_fraction","tension_n"),
        "pressure_contrast_marginal_fraction":
            weighted_fraction(patch,"pressure_contrast_marginal_fraction","pressure_contrast_n"),
        "patch_fraction_nullity_gt_20":float(np.mean(patch.numerical_nullity>20)),
        "patch_fraction_nullity_gt_100":float(np.mean(patch.numerical_nullity>100)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--patch-size",type=int,default=300)
    ap.add_argument("--halo",type=int,default=1)
    ap.add_argument("--rank-rcond",type=float,default=1e-9)
    ap.add_argument("--observable-tol",type=float,default=1e-8)
    ap.add_argument("--marginal-tol",type=float,default=1e-6)
    ap.add_argument("--min-regression-agreement",type=float,default=0.99)
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

    print(
        f"\nUncapped numerical row-space certification "
        f"({a.workers} workers; one pivoted QR per patch)",flush=True
    )

    prec={s:[] for s in SAMPLES}
    vrec={s:[] for s in SAMPLES}
    drec={s:[] for s in SAMPLES}
    done={s:0 for s in SAMPLES}

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs={
            ex.submit(
                one_patch,str(project),s,pid,cells,
                a.rank_rcond,a.observable_tol,a.marginal_tol
            ):(s,pid)
            for s,pid,cells in tasks
        }
        for f in as_completed(futs):
            s,pid,sm,V,D=f.result()
            prec[s].append(sm)
            if len(V):
                V=V.copy();V.insert(0,"patch_id",pid);V.insert(0,"sample",s)
                vrec[s].append(V)
            if len(D):
                D=D.copy();D.insert(0,"patch_id",pid);D.insert(0,"sample",s)
                drec[s].append(D)
            done[s]+=1
            if done[s]==1 or done[s]%50==0 or done[s]==counts[s]:
                print(
                    f"  [{s}] {done[s]}/{counts[s]} "
                    f"nvar={sm['n_variables']} "
                    f"nullity={sm['numerical_nullity']} "
                    f"tau={100*sm['tension_observable_fraction']:.1f}% "
                    f"dp={100*sm['pressure_contrast_observable_fraction']:.1f}%",
                    flush=True
                )

    outroot=project/"results"/"corrections"/"v068"
    outroot.mkdir(parents=True,exist_ok=True)
    reports=[];reg_reports=[];gate_ok=True

    for s in SAMPLES:
        sd=outroot/s;sd.mkdir(parents=True,exist_ok=True)
        P=pd.DataFrame(prec[s]).sort_values("patch_id")
        V=pd.concat(vrec[s],ignore_index=True) if vrec[s] else pd.DataFrame()
        D=pd.concat(drec[s],ignore_index=True) if drec[s] else pd.DataFrame()

        P.to_parquet(sd/"patch_rowspace_observability.parquet",index=False)
        V.to_parquet(sd/"variable_rowspace_observability.parquet",index=False)
        D.to_parquet(sd/"pressure_contrast_rowspace_observability.parquet",index=False)

        T=V[V.variable_class=="tension"].copy()
        tmask=aggregate_global_object_mask(
            T,["object_id"],"tension"
        ) if len(T) else pd.DataFrame()
        dmask=aggregate_global_object_mask(
            D,["interface_id","cell_i","cell_j"],"pressure_contrast"
        ) if len(D) else pd.DataFrame()

        tmask.to_parquet(sd/"tension_production_candidate_mask.parquet",index=False)
        dmask.to_parquet(sd/"pressure_contrast_production_candidate_mask.parquet",index=False)

        sm=summarize_sample(P,V,D)
        sm["sample"]=s
        sm["tension_mask_status_counts"]=(
            tmask.production_candidate_status.value_counts().to_dict() if len(tmask) else {}
        )
        sm["pressure_contrast_mask_status_counts"]=(
            dmask.production_candidate_status.value_counts().to_dict() if len(dmask) else {}
        )

        reg=regression_against_v067(project,s,V,D)
        reg_reports.append(reg)

        vals=[]
        for section in ("variables","pressure_contrasts"):
            x=reg.get(section,{})
            if isinstance(x,dict) and "agreement_fraction" in x and np.isfinite(x["agreement_fraction"]):
                vals.append(float(x["agreement_fraction"]))
        sample_reg_pass=bool(vals and min(vals)>=a.min_regression_agreement)
        sm["v067_regression_pass"]=sample_reg_pass
        sm["v067_min_regression_agreement"]=min(vals) if vals else None
        gate_ok &= sample_reg_pass

        (sd/"summary.json").write_text(json.dumps(sm,indent=2))
        (sd/"v067_regression.json").write_text(json.dumps(reg,indent=2))
        reports.append(sm)

        print(
            f"\n[DONE] {s}: "
            f"tau_obs={100*sm['tension_observable_fraction']:.1f}% "
            f"dp_obs={100*sm['pressure_contrast_observable_fraction']:.1f}% "
            f"checked=100.0% "
            f"v067_agree={sm['v067_min_regression_agreement']}",
            flush=True
        )

    cert={
        "strata_version":"0.6.8.1",
        "stage":"scalable uncapped numerical row-space observability",
        "mechanical_representation":
            "v0.6.4 persistent boundaries + accepted v0.6.5 recovered junctions",
        "biological_geometry_modified":False,
        "mechanics_values_frozen":False,
        "algorithm":{
            "factorization":"row-normalized column-pivoted QR of A^T",
            "one_factorization_per_patch":True,
            "rank_rcond":a.rank_rcond,
            "observable_residual_tolerance":a.observable_tol,
            "marginal_residual_tolerance":a.marginal_tol,
            "dense_nvar_cap":None,
            "nullity_cap":None,
            "pressure_quantities":"cell-cell pressure contrasts; absolute pressures are not promoted by this stage",
        },
        "v067_regression":{
            "minimum_required_agreement":a.min_regression_agreement,
            "reports":reg_reports,
        },
        "sample_reports":reports,
        "ROWSPACE_OBSERVABILITY_GATE":"PASS" if gate_ok else "HOLD",
    }
    (outroot/"rowspace_observability_certificate.json").write_text(
        json.dumps(cert,indent=2)
    )

    print(
        f"\nROWSPACE OBSERVABILITY GATE: "
        f"{cert['ROWSPACE_OBSERVABILITY_GATE']}",
        flush=True
    )
    print(
        f"Certificate: {outroot/'rowspace_observability_certificate.json'}",
        flush=True
    )


if __name__=="__main__":
    main()
