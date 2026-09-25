from __future__ import annotations

import argparse,json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from sutra.hierarchy.v071.resource_intake import sha256_file
from sutra.hierarchy.v077.transport import whiten_covectors
from sutra.hierarchy.v078.holonomy import (
    adjacency_from_registry,
    registry_lookup,
    enumerate_triangles,
    enumerate_chordless_quads,
    evaluate_loops,
    summarize_loop_table,
    certify_loop_table,
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


def one(project,sample,max_quads,coord_checks):
    s1=project/"results"/"hierarchy_v075_local_geometry"/"step1_symmetric_base"/sample
    s3=project/"results"/"hierarchy_v075_local_geometry"/"step3_global_bound"/sample
    v77=project/"results"/"hierarchy_v077_discrete_transport"/sample

    G=sparse.load_npz(require(s1/"symmetric_base_metric.npz")).toarray()
    B=sparse.load_npz(require(s3/"directional_covectors_scaled.npz")).tocsr()
    reg=pd.read_parquet(require(v77/"transport_edge_registry.parquet"))

    # Canonical symmetric whitening inherited from certified v0.7.7.3.
    H,Hinv,Q,rho=whiten_covectors(G,B)

    adj=adjacency_from_registry(reg)
    lookup=registry_lookup(reg)

    triangles=enumerate_triangles(adj)
    quads=enumerate_chordless_quads(adj,max_quads=max_quads)

    tdf=evaluate_loops(
        Q,triangles,lookup,"triangle",coord_checks=coord_checks
    )
    qdf=evaluate_loops(
        Q,quads,lookup,"chordless_quad",coord_checks=coord_checks
    )

    tsum=summarize_loop_table(tdf)
    qsum=summarize_loop_table(qdf)
    tcert=certify_loop_table(tdf)
    qcert=certify_loop_table(qdf)

    all_df=pd.concat([tdf,qdf],ignore_index=True)
    all_sum=summarize_loop_table(all_df)
    all_cert=certify_loop_table(all_df)

    out=project/"results"/"hierarchy_v078_transport_holonomy"/sample
    out.mkdir(parents=True,exist_ok=True)

    writepq(tdf,out/"triangle_holonomy.parquet")
    writepq(qdf,out/"chordless_quad_holonomy.parquet")
    writepq(all_df,out/"all_local_holonomy_loops.parquet")

    figure_cols=[
        "loop_type","loop_index","loop_length",
        "fully_resolved","n_directional_edges","n_identity_edges",
        "active_dimension","identity_deviation_normalized",
        "spectral_angle_rms","spectral_angle_max",
        "reverse_inverse_error",
    ]
    writepq(
        all_df[[c for c in figure_cols if c in all_df.columns]],
        out/"figure_ready_holonomy.parquet"
    )

    # Loop support by number of directional edges.
    support=[]
    if len(all_df):
        for (lt,nrot),g in all_df.groupby(
            ["loop_type","n_directional_edges"],dropna=False
        ):
            support.append({
                "loop_type":str(lt),
                "n_directional_edges":int(nrot),
                "n_loops":int(len(g)),
                "n_resolved":int(g.fully_resolved.sum()),
                "resolved_fraction":float(g.fully_resolved.mean()),
            })
    support_df=pd.DataFrame(support)
    writepq(support_df,out/"holonomy_loop_support.parquet")

    report={
        "sample":sample,
        "n_transport_edges":int(len(reg)),
        "triangle_summary":tsum,
        "quad_summary":qsum,
        "all_loop_summary":all_sum,
        "triangle_certificate":tcert,
        "quad_certificate":qcert,
        "all_loop_certificate":all_cert,
        "policy":{
            "connection_source":"certified v0.7.7 canonical symmetric transport",
            "continuum_curvature_assumed":False,
            "loop_types":["triangle","chordless_quad"],
            "unresolved_edges_repaired":False,
            "holonomy_computed_only_on_fully_resolved_loops":True,
            "identity_edges_contribute_identity":True,
            "ambient_543x543_matrices_materialized":False,
            "reduced_span_dimension_at_most_loop_length":True,
            "quad_cap":int(max_quads),
        },
        "status":"PASS" if all_cert["certificate_pass"] else "HOLD",
    }
    (out/"transport_holonomy_certificate.json").write_text(
        json.dumps(report,indent=2)+"\n"
    )
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--max-quads",type=int,default=20000)
    ap.add_argument("--coordinate-checks",type=int,default=32)
    a=ap.parse_args()

    project=Path(a.project_root).resolve()
    src=require(
        project/"results"/"hierarchy_v077_discrete_transport"/
        "discrete_transport_global_certificate.json"
    )
    d=json.loads(src.read_text())
    if d.get("READY_FOR_TRANSPORT_HOLONOMY") is not True:
        raise SystemExit("ERROR: v0.7.7 transport is not certified for holonomy")

    print("STRATA 0.7.8 | Transport holonomy")
    print("Closed-loop transport on triangles and chordless 4-cycles.")
    print("No continuum curvature tensor is assumed.")
    print("Holonomy is evaluated in the loop directional-frame span only.\n")

    reports=[]
    for s in SAMPLES:
        r=one(project,s,a.max_quads,a.coordinate_checks)
        reports.append(r)
        z=r["all_loop_summary"]
        print(
            f"[DONE] {s}: loops={z['n_candidate_loops']:,} "
            f"resolved={100*z['resolved_fraction']:.1f}% "
            f"directional_resolved={z['n_directional_resolved_loops']:,} "
            f"angle95={z['holonomy_angle_rms_q95']:.4g} "
            f"angle_max={z['holonomy_angle_max_max']:.4g} "
            f"{r['status']}"
        )

    gate=all(r["status"]=="PASS" for r in reports)

    out=project/"results"/"hierarchy_v078_transport_holonomy"
    cert={
        "strata_version":"0.7.8",
        "stage":"transport holonomy on local tissue loops",
        "source_v077_certificate_sha256":sha256_file(src),
        "mathematical_policy":{
            "object":"closed-loop composition of certified discrete transports",
            "curvature_tensor_claimed":False,
            "holonomy_is_operational_connection_observable":True,
            "resolved_loops_only":True,
            "identity_only_loops_required_zero":True,
            "loop_reversal_required_inverse":True,
            "coordinate_relabeling_invariance_audited":True,
        },
        "sample_reports":reports,
        "TRANSPORT_HOLONOMY_GATE":"PASS" if gate else "HOLD",
        "HOLONOMY_CERTIFIED":bool(gate),
        "READY_FOR_DISCRETE_CURVATURE_SUMMARY":bool(gate),
    }
    p=out/"transport_holonomy_global_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print(f"\nTRANSPORT HOLONOMY GATE: {cert['TRANSPORT_HOLONOMY_GATE']}")
    print(f"HOLONOMY CERTIFIED: {cert['HOLONOMY_CERTIFIED']}")
    print(
        "READY FOR DISCRETE CURVATURE SUMMARY: "
        f"{cert['READY_FOR_DISCRETE_CURVATURE_SUMMARY']}"
    )
    print(f"Certificate: {p}")


if __name__=="__main__":
    main()
