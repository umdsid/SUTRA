from __future__ import annotations

import argparse,json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from sutra.hierarchy.v071.resource_intake import sha256_file
from sutra.hierarchy.v075.directional_norm import (
    certify_one_covector,
    empirical_direction_statistics,
    summarize_empirical,
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")


def require(p):
    p=Path(p)
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def writepq(df,path):
    pq.write_table(
        pa.Table.from_pandas(df,preserve_index=False),
        path,
        compression="zstd",
    )


def one(project,sample,n_random):
    s1=project/"results"/"hierarchy_v075_local_geometry"/"step1_symmetric_base"/sample
    s3=project/"results"/"hierarchy_v075_local_geometry"/"step3_global_bound"/sample

    G=sparse.load_npz(require(s1/"symmetric_base_metric.npz")).toarray()
    B=sparse.load_npz(require(s3/"directional_covectors_scaled.npz")).tocsr()

    stats=pd.read_parquet(
        require(s3/"scaled_directional_covector_statistics.parquet")
    )
    dual=stats.scaled_covector_dual_norm.to_numpy(np.float64)

    if B.shape[1] != G.shape[0]:
        raise RuntimeError(
            f"{sample}: covector dimension {B.shape[1]} != G dimension {G.shape[0]}"
        )

    # Exact certificates on the most directionally extreme nodes plus zeros.
    order=np.argsort(dual)
    chosen=set(order[-min(64,len(order)):].tolist())
    chosen.update(order[:min(16,len(order))].tolist())

    certs=[]
    for i in sorted(chosen):
        b=B.getrow(i).toarray().ravel()
        c=certify_one_covector(G,b)
        c["cell_index"]=int(i)
        certs.append(c)

    exact=pd.DataFrame(certs)
    exact_pass=bool(exact.certificate_pass.all()) if len(exact) else False

    empirical=empirical_direction_statistics(
        G,B,dual,n_random=n_random
    )
    summary=summarize_empirical(empirical)

    out=project/"results"/"hierarchy_v075_local_geometry"/"step4_directional_norm"/sample
    out.mkdir(parents=True,exist_ok=True)

    writepq(exact,out/"exact_directional_norm_certificates.parquet")
    writepq(empirical,out/"directional_geometry_node_statistics.parquet")

    fig=empirical[
        [
            "cell_index","dual_norm","positivity_margin",
            "worst_direction_ratio",
            "random_reversal_ratio_median",
            "random_reversal_ratio_max",
            "random_directional_asymmetry_median",
            "random_directional_asymmetry_max",
        ]
    ].copy()
    writepq(fig,out/"figure_ready_directional_geometry.parquet")

    report={
        "sample":sample,
        "construction":"F_x(v)=sqrt(v^T G v)+b_x^T v",
        "public_label":"direction-dependent local geometry",
        "internal_mathematical_class":"Randers-type Finsler norm",
        "n_exact_certified_nodes":int(len(exact)),
        "exact_certificate_pass":exact_pass,
        "empirical_summary":summary,
        "status":"PASS" if (
            exact_pass
            and summary["all_positive_margins"]
            and summary["all_finite"]
        ) else "HOLD",
    }
    (out/"directional_norm_certificate.json").write_text(
        json.dumps(report,indent=2)+"\n"
    )
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--random-directions",type=int,default=8)
    a=ap.parse_args()

    project=Path(a.project_root).resolve()
    source=require(
        project/"results"/"hierarchy_v075_local_geometry"/"step3_global_bound"/
        "global_bound_global_certificate.json"
    )
    d=json.loads(source.read_text())
    if d.get("READY_TO_CONSTRUCT_DIRECTION_DEPENDENT_NORM") is not True:
        raise SystemExit("ERROR: Step 3 did not permit direction-dependent norm")

    print("STRATA 0.7.5 Step 4 | Direction-dependent local geometry")
    print("Constructing F_x(v)=sqrt(v^T G v)+b_x^T v.")
    print("Strict dual bound inherited from Step 3.")
    print("Adversarial exact + deterministic random-direction certification enabled.\n")

    reports=[]
    for sample in SAMPLES:
        r=one(project,sample,a.random_directions)
        reports.append(r)
        s=r["empirical_summary"]
        print(
            f"[DONE] {sample}: directional={100*s['fraction_directional']:.1f}% "
            f"rho_q95={s['dual_norm_q95']:.4f} "
            f"rho_max={s['dual_norm_max']:.4f} "
            f"R95={s['worst_direction_ratio_q95']:.3f} "
            f"Rmax={s['worst_direction_ratio_max']:.3f} "
            f"margin_min={s['positivity_margin_min']:.4f} "
            f"{r['status']}"
        )

    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v075_local_geometry"/"step4_directional_norm"
    cert={
        "strata_version":"0.7.5-step4",
        "stage":"direction-dependent local norm",
        "source_step3_certificate_sha256":sha256_file(source),
        "mathematics":{
            "alpha":"sqrt(v^T G v)",
            "beta":"b_x^T v",
            "local_norm":"alpha+beta",
            "strict_dual_bound":"||b_x||_{G^-1}<1 everywhere",
            "symmetric_limit":"exact when b_x=0",
            "reversal_identity":"F(v)+F(-v)=2 alpha(v)",
        },
        "sample_reports":reports,
        "DIRECTION_DEPENDENT_NORM_GATE":"PASS" if gate else "HOLD",
        "LOCAL_DIRECTIONAL_GEOMETRY_COMPLETE":bool(gate),
        "READY_FOR_GEOMETRY_AWARE_HIERARCHY":bool(gate),
    }
    p=out/"directional_norm_global_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print(f"\nDIRECTION-DEPENDENT NORM GATE: {cert['DIRECTION_DEPENDENT_NORM_GATE']}")
    print(
        "LOCAL DIRECTIONAL GEOMETRY COMPLETE: "
        f"{cert['LOCAL_DIRECTIONAL_GEOMETRY_COMPLETE']}"
    )
    print(
        "READY FOR GEOMETRY-AWARE HIERARCHY: "
        f"{cert['READY_FOR_GEOMETRY_AWARE_HIERARCHY']}"
    )
    print(f"Certificate: {p}")


if __name__=="__main__":
    main()
