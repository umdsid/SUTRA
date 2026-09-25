from __future__ import annotations

import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd,tifffile

from strata_native_mechanics.patches import partition_core_cells,add_halo
from strata_native_mechanics.corrections import (
    CorrectionConfig,classify_background_persistence,compare_patch_corrections
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")


def load_cache(project,s):
    c=project/"results"/"native_mechanics_cache"/s
    E=pd.read_parquet(c/"interfaces.parquet")
    B=pd.read_parquet(c/"background_components.parquet")
    bg=np.load(c/"background_labels.npz")["background_labels"]
    J=pd.read_json(c/"junctions.jsonl",lines=True)
    C=pd.read_parquet(c/"cells.parquet")
    if "incident_interfaces" in J.columns:
        J["incident_interfaces"]=J["incident_interfaces"].apply(
            lambda x:list(x) if isinstance(x,(list,tuple,np.ndarray)) else [])
    return E,B,bg,J,C


def classify_sample(project_s,s,cfg):
    project=Path(project_s)
    E,B,bg,J,C=load_cache(project,s)
    mask=tifffile.imread(
        project/"results"/"production_geometry_r5"/s/"production_segmentation.tif"
    ).astype(np.int32)
    p=classify_background_persistence(mask,bg,B,cfg)
    out=project/"results"/"corrections"/"v064"/s
    out.mkdir(parents=True,exist_ok=True)
    p.to_parquet(out/"mechanical_boundary_persistence.parquet",index=False)
    counts=p.mechanical_boundary_class.value_counts().to_dict()
    return s,counts


def rank_sample(project_s,s,patch_size,halo,cfg):
    project=Path(project_s)
    E,B,bg,J,C=load_cache(project,s)
    p=pd.read_parquet(
        project/"results"/"corrections"/"v064"/s/"mechanical_boundary_persistence.parquet"
    )
    cores=partition_core_cells(E,patch_size=patch_size)
    patches=[add_halo(c,E,hops=halo) for c in cores]
    rows=[]
    for pid,cells in enumerate(patches):
        r=compare_patch_corrections(cells,E,J,C,p,cfg)
        row={"sample":s,"patch_id":pid,"n_patch_cells":len(cells)}
        for mode,q in r.items():
            for k,v in q.items():
                row[f"{mode}__{k}"]=v
        rows.append(row)
    return s,pd.DataFrame(rows)


def summarize(df):
    modes=("baseline","persistent_boundaries","supported_core")
    out={}
    for m in modes:
        null=df[f"{m}__structural_nullity"]
        rank=df[f"{m}__structural_rank_fraction"]
        out[m]={
            "identifiable_patch_fraction":float(np.mean(null==0)),
            "median_structural_nullity":float(np.median(null)),
            "q90_structural_nullity":float(np.quantile(null,.9)),
            "median_rank_fraction":float(np.median(rank)),
        }
        if m!="baseline":
            out[m]["median_nullity_reduction_vs_baseline"]=float(
                np.median(df[f"{m}__nullity_reduction_vs_baseline"])
            )
            out[m]["fraction_patches_improved"]=float(
                np.mean(df[f"{m}__nullity_reduction_vs_baseline"]>0)
            )
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--patch-size",type=int,default=300)
    ap.add_argument("--halo",type=int,default=1)
    ap.add_argument("--unstable-area-max",type=int,default=64)
    ap.add_argument("--unstable-survival-max",type=float,default=.10)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    cfg=CorrectionConfig(
        unstable_area_max=a.unstable_area_max,
        unstable_survival_max=a.unstable_survival_max
    )

    print("Stage A | Mechanical-boundary persistence classification",flush=True)
    with ProcessPoolExecutor(max_workers=min(3,a.workers)) as ex:
        futs={ex.submit(classify_sample,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(futs):
            s,counts=f.result()
            print(f"  [BOUNDARY] {s}: {counts}",flush=True)

    print("\nStage B | Counterfactual rank recovery",flush=True)
    results={}
    with ProcessPoolExecutor(max_workers=min(3,a.workers)) as ex:
        futs={ex.submit(rank_sample,str(project),s,a.patch_size,a.halo,cfg):s for s in SAMPLES}
        for f in as_completed(futs):
            s,df=f.result();results[s]=df
            print(f"  [RANK] {s}: {len(df)} patches complete",flush=True)

    outroot=project/"results"/"corrections"/"v064"
    reports=[]
    for s in SAMPLES:
        df=results[s]
        sdir=outroot/s
        df.to_parquet(sdir/"patch_rank_recovery.parquet",index=False)
        sm=summarize(df)
        p=pd.read_parquet(sdir/"mechanical_boundary_persistence.parquet")
        classes={str(k):int(v) for k,v in p.mechanical_boundary_class.value_counts().to_dict().items()}
        best=max(
            ("baseline","persistent_boundaries","supported_core"),
            key=lambda m:(sm[m]["identifiable_patch_fraction"],
                          sm[m]["median_rank_fraction"])
        )
        report={
            "sample":s,
            "boundary_class_counts":classes,
            "rank_recovery":sm,
            "best_correction_stage":best,
            "mechanics_values_frozen":False,
        }
        (sdir/"summary.json").write_text(json.dumps(report,indent=2))
        reports.append(report)
        print(
            f"\n[DONE] {s}: "
            f"ident baseline={100*sm['baseline']['identifiable_patch_fraction']:.1f}% "
            f"persistent={100*sm['persistent_boundaries']['identifiable_patch_fraction']:.1f}% "
            f"supported_core={100*sm['supported_core']['identifiable_patch_fraction']:.1f}% "
            f"best={best}",flush=True
        )

    cert={
        "strata_version":"0.6.4",
        "stage":"mechanical corrections / rank recovery",
        "corrections_directory":str(outroot),
        "frozen_biological_geometry_modified":False,
        "mechanics_values_frozen":False,
        "corrections":[
            "morphological persistence classification of background components",
            "suppression of unstable micro-gaps from mechanics-only boundary topology",
            "supported-mechanics core excluding weak-curvature interfaces lacking junction support",
        ],
        "sample_reports":reports,
    }
    (outroot/"corrections_certificate.json").write_text(json.dumps(cert,indent=2))
    print(f"\nCertificate: {outroot/'corrections_certificate.json'}",flush=True)

if __name__=="__main__":
    main()
