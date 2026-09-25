from __future__ import annotations
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd,tifffile
from sutra.vmsi_compat.audit import (
    disconnected,boundary_relabel_like_readme,faq_cleanup_like_readme,
    mapped_component_audit,source_manifest
)

def one(project_s,sample):
    project=Path(project_s);tm=project/"external"/"TensionMap"
    src=project/"results"/"production_geometry_r5_connectivity"/sample
    mask=tifffile.imread(src/"production_segmentation_connectivity.tif").astype(np.int32)
    original_bad=disconnected(mask,1)
    out=project/"results"/"vmsi_compatibility_audit"/sample
    out.mkdir(parents=True,exist_ok=True)

    base8=disconnected(mask,2)
    quick=boundary_relabel_like_readme(mask)
    qbad4=disconnected(quick,1);qbad8=disconnected(quick,2)

    pd.DataFrame(
        mapped_component_audit(mask,quick,original_bad,subpixel=True)
    ).to_csv(out/"original_failed_to_boundary_relabel.csv",index=False)

    cleanup_rows=[]
    for min_size in (2,5,10,20):
        clean=faq_cleanup_like_readme(mask,min_size=min_size,expand_distance=5)
        c4=disconnected(clean,1);c8=disconnected(clean,2)
        cleanup_rows.append({
            "min_size":min_size,"expand_distance":5,
            "n_labels":int(clean.max()),
            "disconnected_4":len(c4),"disconnected_8":len(c8),
            "changed_pixel_fraction":float(np.mean(clean!=mask)),
        })
        pd.DataFrame(
            mapped_component_audit(mask,clean,original_bad,subpixel=False)
        ).to_csv(out/f"original_failed_to_faq_cleanup_min{min_size}.csv",index=False)
    pd.DataFrame(cleanup_rows).to_csv(out/"faq_cleanup_sweep.csv",index=False)

    orig4=set(original_bad);orig8=set(base8)
    report={
        "sample":sample,
        "source_manifest":source_manifest(tm),
        "input":{
            "n_labels":int(mask.max()),"shape":list(mask.shape),
            "disconnected_4":len(original_bad),
            "disconnected_8":len(base8),
            "four_only_labels":len(orig4-orig8),
            "four_only_examples":sorted(orig4-orig8)[:30],
        },
        "quickstart_boundary_relabel":{
            "n_labels":int(quick.max()),
            "shape":list(quick.shape),
            "disconnected_4":len(qbad4),
            "disconnected_8":len(qbad8),
            "grid_relation":"subpixel grid: (2H-1,2W-1)",
        },
        "faq_cleanup_sweep":cleanup_rows,
    }
    (out/"vmsi_compatibility_report.json").write_text(json.dumps(report,indent=2))
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--project-root",default=".")
    a=ap.parse_args();project=Path(a.project_root).resolve()
    samples=["alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"];rs=[]
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs={ex.submit(one,str(project),s):s for s in samples}
        for f in as_completed(futs):
            r=f.result();rs.append(r)
            print(
                f"[DONE] {r['sample']}: "
                f"STRATA 4c={r['input']['disconnected_4']} "
                f"8c={r['input']['disconnected_8']} "
                f"4-only={r['input']['four_only_labels']} | "
                f"README relabel 4c={r['quickstart_boundary_relabel']['disconnected_4']} "
                f"shape={tuple(r['quickstart_boundary_relabel']['shape'])}"
            )
            for q in r["faq_cleanup_sweep"]:
                print(f"    FAQ min={q['min_size']}: 4c={q['disconnected_4']} 8c={q['disconnected_8']} changed={100*q['changed_pixel_fraction']:.3f}%")
    rs.sort(key=lambda x:samples.index(x["sample"]))
    out=project/"results"/"vmsi_compatibility_audit";out.mkdir(parents=True,exist_ok=True)
    cert={"strata_version":"0.5.14","stage":"VMSI representation compatibility audit","read_only":True,"sample_reports":rs}
    (out/"vmsi_compatibility_certificate.json").write_text(json.dumps(cert,indent=2))
    print(f"\nCertificate: {out/'vmsi_compatibility_certificate.json'}")
if __name__=="__main__":main()
