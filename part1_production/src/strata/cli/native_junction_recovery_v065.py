from __future__ import annotations

import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd,tifffile

from strata_native_mechanics.patches import partition_core_cells,add_halo
from strata_native_mechanics.corrections import (
    CorrectionConfig,corrected_objects,patch_rank
)
from strata_native_mechanics.junction_recovery import (
    JunctionRecoveryConfig,recover_microgap_junctions,
    patch_recovered_junction_count
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


def prepare_sample(project_s,s,jcfg):
    project=Path(project_s)
    E,B,bg,J,C=load_cache(project,s)
    P=pd.read_parquet(
        project/"results"/"corrections"/"v064"/s/"mechanical_boundary_persistence.parquet"
    )
    mask=tifffile.imread(
        project/"results"/"production_geometry_r5"/s/"production_segmentation.tif"
    ).astype(np.int32)

    # Freeze the v0.6.4 winner as the starting mechanics representation.
    Ep,Jp,_=corrected_objects(E,J,P,CorrectionConfig(),"persistent_boundaries")
    Jr,detail,audit=recover_microgap_junctions(mask,bg,P,Ep,Jp,jcfg)

    out=project/"results"/"corrections"/"v065"/s
    out.mkdir(parents=True,exist_ok=True)
    detail.to_csv(out/"recovered_junctions.csv",index=False)
    # JSONL handles the list-valued incident_interfaces safely.
    Jr.to_json(out/"junctions_persistent_plus_recovered.jsonl",
               orient="records",lines=True)
    (out/"junction_recovery_audit.json").write_text(json.dumps(audit,indent=2))
    return s,audit


def rank_sample(project_s,s,patch_size,halo):
    project=Path(project_s)
    E,B,bg,J,C=load_cache(project,s)
    P=pd.read_parquet(
        project/"results"/"corrections"/"v064"/s/"mechanical_boundary_persistence.parquet"
    )
    Ep,Jp,_=corrected_objects(E,J,P,CorrectionConfig(),"persistent_boundaries")
    Jr=pd.read_json(
        project/"results"/"corrections"/"v065"/s/"junctions_persistent_plus_recovered.jsonl",
        lines=True
    )
    if "incident_interfaces" in Jr.columns:
        Jr["incident_interfaces"]=Jr["incident_interfaces"].apply(
            lambda x:list(x) if isinstance(x,(list,tuple,np.ndarray)) else [])
    detail_path=project/"results"/"corrections"/"v065"/s/"recovered_junctions.csv"
    detail=pd.read_csv(detail_path) if detail_path.exists() and detail_path.stat().st_size else pd.DataFrame()

    cores=partition_core_cells(E,patch_size=patch_size)
    patches=[add_halo(c,E,hops=halo) for c in cores]
    rows=[]
    for pid,cells in enumerate(patches):
        a=patch_rank(cells,Ep,Jp,C)
        b=patch_rank(cells,Ep,Jr,C)
        rows.append({
            "sample":s,"patch_id":pid,"n_patch_cells":len(cells),
            "recovered_junctions_in_patch":
                patch_recovered_junction_count(cells,detail,Ep),
            "persistent__nullity":a["structural_nullity"],
            "persistent__rank_fraction":a["structural_rank_fraction"],
            "persistent__identifiable":a["structurally_identifiable"],
            "persistent__junction_rows":a["n_junction_rows"],
            "recovered__nullity":b["structural_nullity"],
            "recovered__rank_fraction":b["structural_rank_fraction"],
            "recovered__identifiable":b["structurally_identifiable"],
            "recovered__junction_rows":b["n_junction_rows"],
            "nullity_reduction":
                a["structural_nullity"]-b["structural_nullity"],
            "rank_fraction_gain":
                b["structural_rank_fraction"]-a["structural_rank_fraction"],
        })
    return s,pd.DataFrame(rows)


def summarize(df):
    return {
        "persistent_identifiable_patch_fraction":
            float(np.mean(df.persistent__nullity==0)),
        "recovered_identifiable_patch_fraction":
            float(np.mean(df.recovered__nullity==0)),
        "persistent_median_nullity":
            float(np.median(df.persistent__nullity)),
        "recovered_median_nullity":
            float(np.median(df.recovered__nullity)),
        "persistent_q90_nullity":
            float(np.quantile(df.persistent__nullity,.9)),
        "recovered_q90_nullity":
            float(np.quantile(df.recovered__nullity,.9)),
        "fraction_patches_rank_improved":
            float(np.mean(df.nullity_reduction>0)),
        "fraction_patches_with_recovered_junction":
            float(np.mean(df.recovered_junctions_in_patch>0)),
        "total_patch_junction_incidence":
            int(df.recovered_junctions_in_patch.sum()),
        "median_nullity_reduction_on_affected_patches":
            float(np.median(df.loc[df.recovered_junctions_in_patch>0,
                                   "nullity_reduction"]))
            if np.any(df.recovered_junctions_in_patch>0) else 0.0,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--patch-size",type=int,default=300)
    ap.add_argument("--halo",type=int,default=1)
    ap.add_argument("--local-pad",type=int,default=5)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    jcfg=JunctionRecoveryConfig(local_pad=a.local_pad)

    print("STRATA 0.6.5 | Persistent boundaries + conservative junction recovery")
    print("Frozen biological geometry remains read-only.\n",flush=True)

    print("Stage A | Unique local junction recovery",flush=True)
    audits={}
    with ProcessPoolExecutor(max_workers=min(3,a.workers)) as ex:
        futs={ex.submit(prepare_sample,str(project),s,jcfg):s for s in SAMPLES}
        for f in as_completed(futs):
            s,audit=f.result();audits[s]=audit
            print(
                f"  [JUNCTION] {s}: "
                f"unstable={sum(audit.values()) if audit else 0} "
                f"three_cells={audit.get('exactly_three_cells',0)} "
                f"pairwise={audit.get('all_pairwise_interfaces_exist',0)} "
                f"local={audit.get('local_pairwise_contacts_exist',0)} "
                f"recovered={audit.get('recovered',0)}",
                flush=True
            )

    print("\nStage B | Rank recovery",flush=True)
    dfs={}
    with ProcessPoolExecutor(max_workers=min(3,a.workers)) as ex:
        futs={ex.submit(rank_sample,str(project),s,a.patch_size,a.halo):s
              for s in SAMPLES}
        for f in as_completed(futs):
            s,df=f.result();dfs[s]=df
            print(f"  [RANK] {s}: {len(df)} patches complete",flush=True)

    outroot=project/"results"/"corrections"/"v065"
    reports=[]
    for s in SAMPLES:
        df=dfs[s]
        sdir=outroot/s
        df.to_parquet(sdir/"junction_rank_recovery.parquet",index=False)
        sm=summarize(df)
        report={
            "sample":s,
            "junction_recovery_audit":audits[s],
            "rank_recovery":sm,
            "frozen_biological_geometry_modified":False,
            "mechanics_values_frozen":False,
        }
        (sdir/"summary.json").write_text(json.dumps(report,indent=2))
        reports.append(report)
        print(
            f"\n[DONE] {s}: ident persistent="
            f"{100*sm['persistent_identifiable_patch_fraction']:.1f}% "
            f"-> recovered={100*sm['recovered_identifiable_patch_fraction']:.1f}% "
            f"patches_improved={100*sm['fraction_patches_rank_improved']:.1f}% "
            f"recovered_junctions={audits[s].get('recovered',0)}",
            flush=True
        )

    cert={
        "strata_version":"0.6.5",
        "stage":"persistent-boundary + junction-recovery correction",
        "corrections_directory":str(outroot),
        "recovery_policy":{
            "exactly_three_touching_cells":True,
            "all_three_measured_pairwise_interfaces_required":True,
            "all_three_pairwise_contacts_must_be_local":True,
            "nondegenerate_angular_geometry_required":True,
            "new_tension_variables_created":False,
        },
        "sample_reports":reports,
        "mechanics_values_frozen":False,
    }
    (outroot/"corrections_certificate.json").write_text(json.dumps(cert,indent=2))
    print(f"\nCertificate: {outroot/'corrections_certificate.json'}",flush=True)

if __name__=="__main__":
    main()
