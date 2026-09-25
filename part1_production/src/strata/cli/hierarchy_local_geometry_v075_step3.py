from __future__ import annotations
import argparse,json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from strata_hierarchy.v071.resource_intake import sha256_file
from strata_hierarchy.v075.bound_calibration import (
    global_scale,apply_global_scale,scaled_dual_norms,
    certify_scaled_norms,sensitivity_table,sparse_sha256
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def require(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p

def writepq(df,path):
    pq.write_table(
        pa.Table.from_pandas(df,preserve_index=False),
        path,compression="zstd"
    )

def load_sample(project,sample):
    s2=project/"results"/"hierarchy_v075_local_geometry"/"step2_directional_covector"/sample
    cert=json.loads(require(s2/"directional_covector_certificate.json").read_text())
    B=sparse.load_npz(require(s2/"raw_directional_covectors.npz")).tocsr()
    stats=pd.read_parquet(require(s2/"raw_directional_covector_statistics.parquet"))
    d=stats.raw_covector_dual_norm.to_numpy(np.float64)
    if sparse_sha256(B)!=cert["raw_covector_sha256"]:
        raise RuntimeError(f"{sample}: raw covector hash mismatch")
    return cert,B,stats,d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--target-max-dual-norm",type=float,default=0.90)
    a=ap.parse_args()

    project=Path(a.project_root).resolve()
    source=require(
        project/"results"/"hierarchy_v075_local_geometry"/"step2_directional_covector"/
        "directional_covector_global_certificate.json"
    )
    src=json.loads(source.read_text())
    if src.get("READY_FOR_GLOBAL_BOUND_CALIBRATION") is not True:
        raise SystemExit("ERROR: Step 2 did not permit global calibration")

    target=float(a.target_max_dual_norm)
    global_raw=float(src["global_raw_dual_norm_max"])
    kappa=global_scale(global_raw,target)

    print("STRATA 0.7.5 Step 3 | Global directional-bound calibration")
    print(f"Raw global dual-norm maximum: {global_raw:.9g}")
    print(f"Shared target maximum: {target:.3f}")
    print(f"Shared scale kappa: {kappa:.9g}")
    print("No local clipping. No specimen-specific scaling.\n")

    loaded={}
    raw_by_sample={}
    for sample in SAMPLES:
        cert,B,stats,d=load_sample(project,sample)
        loaded[sample]=(cert,B,stats,d)
        raw_by_sample[sample]=d

    # Fixed diagnostic sweep; this does not alter the production target.
    targets=(0.50,0.70,0.80,0.90,0.95)
    sens=sensitivity_table(raw_by_sample,global_raw,targets)

    outroot=project/"results"/"hierarchy_v075_local_geometry"/"step3_global_bound"
    outroot.mkdir(parents=True,exist_ok=True)
    writepq(sens,outroot/"global_bound_sensitivity.parquet")

    reports=[]
    for sample in SAMPLES:
        cert,B,stats,d=loaded[sample]
        Bs=apply_global_scale(B,kappa)
        ds=scaled_dual_norms(d,kappa)
        c=certify_scaled_norms(ds,target)

        out=outroot/sample
        out.mkdir(parents=True,exist_ok=True)
        sparse.save_npz(out/"directional_covectors_scaled.npz",Bs,compressed=True)

        st=stats.copy()
        st["scaled_covector_dual_norm"]=ds
        st["finsler_positivity_margin"]=1.0-ds
        writepq(st,out/"scaled_directional_covector_statistics.parquet")

        report={
            "sample":sample,
            "raw_covector_sha256":cert["raw_covector_sha256"],
            "scaled_covector_sha256":sparse_sha256(Bs),
            "global_scale_kappa":kappa,
            "target_global_max_dual_norm":target,
            "scaling_shared_across_specimens":True,
            "local_clipping":False,
            "specimen_specific_scaling":False,
            "relative_raw_magnitudes_preserved":True,
            "certification":c,
            "status":"PASS" if c["certificate_pass"] else "HOLD",
        }
        (out/"global_bound_certificate.json").write_text(
            json.dumps(report,indent=2)+"\n"
        )
        reports.append(report)

        print(
            f"[DONE] {sample}: max={c['scaled_dual_norm_max']:.6f} "
            f"margin={c['minimum_finsler_positivity_margin']:.6f} "
            f"Rmax<={c['worst_case_reversibility_bound']:.3f} "
            f"{report['status']}"
        )

    gate=all(r["status"]=="PASS" for r in reports)
    # At least one specimen must attain the global target to verify that the
    # global normalization is actually based on the supplied maximum.
    attained=max(
        r["certification"]["scaled_dual_norm_max"] for r in reports
    )

    certout={
        "strata_version":"0.7.5-step3",
        "stage":"global directional-bound calibration",
        "source_step2_certificate_sha256":sha256_file(source),
        "global_raw_dual_norm_max":global_raw,
        "target_global_max_dual_norm":target,
        "global_scale_kappa":kappa,
        "observed_global_scaled_dual_norm_max":attained,
        "policy":{
            "one_scale_all_specimens":True,
            "local_clipping":False,
            "specimen_specific_scaling":False,
            "relative_directional_magnitudes_preserved":True,
            "sensitivity_targets":[0.50,0.70,0.80,0.90,0.95],
        },
        "sample_reports":reports,
        "GLOBAL_BOUND_GATE":"PASS" if gate else "HOLD",
        "READY_TO_CONSTRUCT_DIRECTION_DEPENDENT_NORM":bool(gate),
    }
    p=outroot/"global_bound_global_certificate.json"
    p.write_text(json.dumps(certout,indent=2)+"\n")

    print(f"\nGLOBAL BOUND GATE: {certout['GLOBAL_BOUND_GATE']}")
    print(
        "READY TO CONSTRUCT DIRECTION-DEPENDENT NORM: "
        f"{certout['READY_TO_CONSTRUCT_DIRECTION_DEPENDENT_NORM']}"
    )
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
