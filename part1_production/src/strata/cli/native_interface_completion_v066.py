from __future__ import annotations
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd,tifffile
from strata_native_mechanics.patches import partition_core_cells,add_halo
from strata_native_mechanics.corrections import CorrectionConfig,corrected_objects,patch_rank
from strata_native_mechanics.interface_completion import InterfaceCompletionConfig,complete_interfaces_and_junctions

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def load(project,s):
    c=project/"results"/"native_mechanics_cache"/s
    E=pd.read_parquet(c/"interfaces.parquet")
    B=pd.read_parquet(c/"background_components.parquet")
    bg=np.load(c/"background_labels.npz")["background_labels"]
    J=pd.read_json(c/"junctions.jsonl",lines=True)
    C=pd.read_parquet(c/"cells.parquet")
    if "incident_interfaces" in J.columns:
        J["incident_interfaces"]=J["incident_interfaces"].apply(lambda x:list(x) if isinstance(x,(list,tuple,np.ndarray)) else [])
    return E,B,bg,J,C

def prep(project_s,s,cfg):
    project=Path(project_s);E,B,bg,J,C=load(project,s)
    P=pd.read_parquet(project/"results"/"corrections"/"v064"/s/"mechanical_boundary_persistence.parquet")
    mask=tifffile.imread(project/"results"/"production_geometry_r5"/s/"production_segmentation.tif").astype(np.int32)
    Ep,Jp,_=corrected_objects(E,J,P,CorrectionConfig(),"persistent_boundaries")
    Ec,Jc,D,A=complete_interfaces_and_junctions(mask,bg,P,Ep,Jp,cfg)
    out=project/"results"/"corrections"/"v066"/s;out.mkdir(parents=True,exist_ok=True)
    Ec.to_parquet(out/"interfaces_completed.parquet",index=False)
    Jc.to_json(out/"junctions_completed.jsonl",orient="records",lines=True)
    D.to_csv(out/"accepted_completions.csv",index=False)
    (out/"completion_audit.json").write_text(json.dumps(A,indent=2))
    return s,A

def rank_sample(project_s,s,patch_size,halo):
    project=Path(project_s);E,B,bg,J,C=load(project,s)
    P=pd.read_parquet(project/"results"/"corrections"/"v064"/s/"mechanical_boundary_persistence.parquet")
    Ep,Jp,_=corrected_objects(E,J,P,CorrectionConfig(),"persistent_boundaries")
    Ec=pd.read_parquet(project/"results"/"corrections"/"v066"/s/"interfaces_completed.parquet")
    Jc=pd.read_json(project/"results"/"corrections"/"v066"/s/"junctions_completed.jsonl",lines=True)
    if "incident_interfaces" in Jc.columns:
        Jc["incident_interfaces"]=Jc["incident_interfaces"].apply(lambda x:list(x) if isinstance(x,(list,tuple,np.ndarray)) else [])
    cores=partition_core_cells(E,patch_size=patch_size)
    patches=[add_halo(c,E,hops=halo) for c in cores]
    rows=[]
    for pid,cells in enumerate(patches):
        a=patch_rank(cells,Ep,Jp,C);b=patch_rank(cells,Ec,Jc,C)
        rows.append({
            "sample":s,"patch_id":pid,"n_patch_cells":len(cells),
            "persistent_nullity":a["structural_nullity"],
            "completed_nullity":b["structural_nullity"],
            "persistent_rank_fraction":a["structural_rank_fraction"],
            "completed_rank_fraction":b["structural_rank_fraction"],
            "nullity_reduction":a["structural_nullity"]-b["structural_nullity"],
        })
    return s,pd.DataFrame(rows)

def summary(df):
    return {
        "persistent_identifiable_fraction":float(np.mean(df.persistent_nullity==0)),
        "completed_identifiable_fraction":float(np.mean(df.completed_nullity==0)),
        "persistent_median_nullity":float(np.median(df.persistent_nullity)),
        "completed_median_nullity":float(np.median(df.completed_nullity)),
        "persistent_q90_nullity":float(np.quantile(df.persistent_nullity,.9)),
        "completed_q90_nullity":float(np.quantile(df.completed_nullity,.9)),
        "fraction_patches_improved":float(np.mean(df.nullity_reduction>0)),
        "median_nullity_reduction_improved":float(np.median(df.loc[df.nullity_reduction>0,"nullity_reduction"])) if np.any(df.nullity_reduction>0) else 0.0,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--patch-size",type=int,default=300)
    ap.add_argument("--halo",type=int,default=1)
    a=ap.parse_args();project=Path(a.project_root).resolve()
    cfg=InterfaceCompletionConfig()

    print("STRATA 0.6.6 | Counterfactual micro-gap interface completion")
    print("Rank-only. Frozen biological geometry remains read-only.\n")
    audits={}
    with ProcessPoolExecutor(max_workers=min(3,a.workers)) as ex:
        futs={ex.submit(prep,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(futs):
            s,A=f.result();audits[s]=A
            print(f"[COMPLETE] {s}: three_cells={A.get('three_cells',0)} "
                  f"one_missing={A.get('one_missing_pair',0)} "
                  f"supported={A.get('supported_virtual_contact',0)} "
                  f"accepted={A.get('accepted',0)}",flush=True)

    dfs={}
    with ProcessPoolExecutor(max_workers=min(3,a.workers)) as ex:
        futs={ex.submit(rank_sample,str(project),s,a.patch_size,a.halo):s for s in SAMPLES}
        for f in as_completed(futs):
            s,df=f.result();dfs[s]=df
            print(f"[RANK] {s}: {len(df)} patches",flush=True)

    root=project/"results"/"corrections"/"v066";reports=[]
    for s in SAMPLES:
        df=dfs[s];sd=root/s;df.to_parquet(sd/"rank_recovery.parquet",index=False)
        sm=summary(df)
        rep={"sample":s,"completion_audit":audits[s],"rank_recovery":sm,
             "mechanics_values_frozen":False}
        (sd/"summary.json").write_text(json.dumps(rep,indent=2));reports.append(rep)
        print(f"\n[DONE] {s}: ident {100*sm['persistent_identifiable_fraction']:.1f}% "
              f"-> {100*sm['completed_identifiable_fraction']:.1f}% "
              f"patches_improved={100*sm['fraction_patches_improved']:.1f}% "
              f"accepted={audits[s].get('accepted',0)}",flush=True)

    cert={"strata_version":"0.6.6","stage":"counterfactual interface completion",
          "rank_only":True,"biological_geometry_modified":False,
          "virtual_interface_curvature_confidence":0.0,
          "sample_reports":reports}
    (root/"corrections_certificate.json").write_text(json.dumps(cert,indent=2))
    print(f"\nCertificate: {root/'corrections_certificate.json'}")

if __name__=="__main__":main()
