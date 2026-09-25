from __future__ import annotations

import argparse,json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from strata_hierarchy.v071.resource_intake import sha256_file
from strata_hierarchy.v077.transport import (
    whiten_covectors,
    classify_pair,
    transport_audit_one,
)
from strata_hierarchy.v077.geodesics import (
    build_directed_cost_graph,
    graph_component_summary,
    choose_landmarks,
    landmark_geodesics,
    triangle_audit,
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


def one(project,sample,n_landmarks,audit_edges):
    s1=project/"results"/"hierarchy_v075_local_geometry"/"step1_symmetric_base"/sample
    s3=project/"results"/"hierarchy_v075_local_geometry"/"step3_global_bound"/sample
    v76=project/"results"/"hierarchy_v076_geometry_aware"/sample
    v762=project/"results"/"hierarchy_v0762_pressure_field"/sample

    G=sparse.load_npz(require(s1/"symmetric_base_metric.npz")).toarray()
    B=sparse.load_npz(require(s3/"directional_covectors_scaled.npz")).tocsr()
    level=pd.read_parquet(require(v76/"boundary_geometry_all_levels.parquet"))
    level1=level[level.level==1].copy()

    # Pressure finalization is a hard dependency even though the already-frozen
    # v0.7.6 edge costs are reused here. This prevents transport from proceeding
    # on a pressure field that has not passed the potential/residual contract.
    pcert=json.loads(require(v762/"pressure_field_certificate.json").read_text())
    if pcert.get("status")!="PASS":
        raise RuntimeError(f"{sample}: pressure field is not finalized")

    H,Hinv,Q,rho=whiten_covectors(G,B)

    geom=level1[
        level1.geometry_state_resolved.astype(bool)
        & np.isfinite(level1.local_forward_cost)
        & np.isfinite(level1.local_reverse_cost)
    ].copy()

    # Compact per-edge transport metadata. No 543x543 matrices are materialized.
    rows=[]
    for r in geom.itertuples(index=False):
        u=int(r.super_i);v=int(r.super_j)
        info=classify_pair(Q[u],Q[v])
        rows.append({
            "super_i":u,
            "super_j":v,
            "local_forward_cost":float(r.local_forward_cost),
            "local_reverse_cost":float(r.local_reverse_cost),
            "geometry_pair_cost":float(r.geometry_pair_cost),
            **info,
        })
    reg=pd.DataFrame(rows)

    # Deterministic exact operator audit: prioritize actual directional rotations,
    # then identity edges.
    rot=reg[reg.transport_status=="DIRECTIONAL_ROTATION"]
    ident=reg[reg.transport_status=="IDENTITY_SYMMETRIC"]
    audit=pd.concat([
        rot.head(min(audit_edges,len(rot))),
        ident.head(min(64,len(ident))),
    ],ignore_index=True)

    arows=[]
    for k,r in enumerate(audit.itertuples(index=False)):
        c=transport_audit_one(
            G,H,Hinv,Q[int(r.super_i)],Q[int(r.super_j)],seed=1000+k
        )
        arows.append({
            "super_i":int(r.super_i),
            "super_j":int(r.super_j),
            **c,
        })
    audit_df=pd.DataFrame(arows)

    # Directed geodesic graph uses all geometry-resolved Level-0 edges, not
    # hierarchy admissibility; the latter is a contraction rule, not a movement rule.
    A,cost_edges=build_directed_cost_graph(geom,B.shape[0])
    comp_labels,comp_summary=graph_component_summary(A)
    landmarks=choose_landmarks(rho,comp_labels,n_landmarks)
    geo_df,geo_summary=landmark_geodesics(A,landmarks)
    tri=triangle_audit(A,landmarks)

    status_counts=reg.transport_status.value_counts().to_dict()
    n_edges=len(reg)
    rotation_fraction=float(
        status_counts.get("DIRECTIONAL_ROTATION",0)/n_edges
    ) if n_edges else 0.0
    identity_fraction=float(
        status_counts.get("IDENTITY_SYMMETRIC",0)/n_edges
    ) if n_edges else 0.0
    unresolved_fraction=float(
        (
            status_counts.get("ONE_SIDED_UNRESOLVED",0)
            +status_counts.get("ANTIPODAL_UNRESOLVED",0)
        )/n_edges
    ) if n_edges else 0.0

    exact_pass=bool(
        len(audit_df)==0 or audit_df.certificate_pass.astype(bool).all()
    )
    costs_positive=bool(
        (geom.local_forward_cost>=0).all()
        and (geom.local_reverse_cost>=0).all()
    )
    geodesic_pass=bool(tri["pass"] and costs_positive)

    out=project/"results"/"hierarchy_v077_discrete_transport"/sample
    out.mkdir(parents=True,exist_ok=True)
    writepq(reg,out/"transport_edge_registry.parquet")
    writepq(audit_df,out/"transport_operator_audit.parquet")
    writepq(
        pd.DataFrame({
            "cell_index":np.arange(len(rho),dtype=np.int64),
            "directional_dual_norm":rho,
            "transport_component":comp_labels.astype(np.int64),
        }),
        out/"transport_node_registry.parquet"
    )
    writepq(geo_df,out/"landmark_directed_geodesics.parquet")
    writepq(
        pd.DataFrame({"cell_index":landmarks}),
        out/"geodesic_landmarks.parquet"
    )

    report={
        "sample":sample,
        "n_geometry_resolved_level0_edges":int(n_edges),
        "transport_status_counts":{str(k):int(v) for k,v in status_counts.items()},
        "rotation_fraction":rotation_fraction,
        "symmetric_identity_fraction":identity_fraction,
        "unresolved_directional_frame_fraction":unresolved_fraction,
        "operator_audit":{
            "n_edges_checked":int(len(audit_df)),
            "all_certified":exact_pass,
            "max_alpha_isometry_error":
                float(audit_df.alpha_isometry_error.dropna().max())
                if len(audit_df) and audit_df.alpha_isometry_error.notna().any()
                else 0.0,
            "max_direction_map_error":
                float(audit_df.direction_map_error.dropna().max())
                if len(audit_df) and audit_df.direction_map_error.notna().any()
                else 0.0,
            "max_roundtrip_error":
                float(audit_df.roundtrip_error.dropna().max())
                if len(audit_df) and audit_df.roundtrip_error.notna().any()
                else 0.0,
        },
        "transport_graph":comp_summary,
        "directed_geodesics":geo_summary,
        "triangle_audit":tri,
        "pressure_contract_verified":True,
        "connection_policy":{
            "declared_discrete_connection":True,
            "symmetric_metric_isometry":True,
            "preferred_direction_alignment":True,
            "directional_strength_preservation":False,
            "directional_strength_mismatch_recorded":True,
            "one_sided_direction_not_fabricated":True,
            "antipodal_case_not_fabricated":True,
            "full_transport_matrices_materialized":False,
        },
        "status":"PASS" if exact_pass and geodesic_pass else "HOLD",
    }
    (out/"discrete_transport_certificate.json").write_text(
        json.dumps(report,indent=2)+"\n"
    )
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--landmarks",type=int,default=16)
    ap.add_argument("--audit-edges",type=int,default=256)
    a=ap.parse_args()

    project=Path(a.project_root).resolve()

    geometry=require(
        project/"results"/"hierarchy_v076_geometry_aware"/
        "geometry_aware_hierarchy_certificate.json"
    )
    gd=json.loads(geometry.read_text())
    if gd.get("READY_FOR_DISCRETE_TRANSPORT") is not True:
        raise SystemExit("ERROR: v0.7.6 geometry hierarchy did not permit transport")

    pressure=require(
        project/"results"/"hierarchy_v0762_pressure_field"/
        "pressure_field_global_certificate.json"
    )
    pd0=json.loads(pressure.read_text())
    if pd0.get("READY_FOR_DISCRETE_TRANSPORT") is not True:
        raise SystemExit("ERROR: pressure field is not frozen for transport")

    print("STRATA 0.7.7 | Discrete directional transport")
    print("Explicit minimal-frame discrete connection; no implicit Euclidean connection.")
    print("Directed geodesics use the frozen local forward/reverse edge costs.")
    print("One-sided and antipodal directional frames remain unresolved.\n")

    reports=[]
    for s in SAMPLES:
        r=one(project,s,a.landmarks,a.audit_edges)
        reports.append(r)
        g=r["directed_geodesics"]
        print(
            f"[DONE] {s}: edges={r['n_geometry_resolved_level0_edges']:,} "
            f"rotate={100*r['rotation_fraction']:.1f}% "
            f"identity={100*r['symmetric_identity_fraction']:.1f}% "
            f"unresolved={100*r['unresolved_directional_frame_fraction']:.1f}% "
            f"components={r['transport_graph']['n_components']:,} "
            f"geo_R95={g['distance_asymmetry_ratio_q95']:.3f} "
            f"{r['status']}"
        )

    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v077_discrete_transport"
    cert={
        "strata_version":"0.7.7",
        "stage":"discrete directional transport + landmark geodesics",
        "source_geometry_certificate_sha256":sha256_file(geometry),
        "source_pressure_certificate_sha256":sha256_file(pressure),
        "mathematical_policy":{
            "local_norm_alone_assumed_to_define_connection":False,
            "connection_declared_explicitly":True,
            "connection":"minimal G-isometric rotation using canonical symmetric metric square root",
            "identity_on_symmetric_zero-zero_edges":True,
            "one_sided_directional_frame":"unresolved",
            "antipodal_directional_frame":"unresolved",
            "directed_geodesic_costs":"frozen v0.7.6 local forward/reverse costs",
            "contraction_admissibility_used_as_transport_gate":False,
        },
        "sample_reports":reports,
        "DISCRETE_TRANSPORT_GATE":"PASS" if gate else "HOLD",
        "DISCRETE_CONNECTION_CERTIFIED":bool(gate),
        "DIRECTED_GEODESICS_CERTIFIED":bool(gate),
        "READY_FOR_TRANSPORT_HOLONOMY":bool(gate),
    }
    p=out/"discrete_transport_global_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print(f"\nDISCRETE TRANSPORT GATE: {cert['DISCRETE_TRANSPORT_GATE']}")
    print(f"DISCRETE CONNECTION CERTIFIED: {cert['DISCRETE_CONNECTION_CERTIFIED']}")
    print(f"DIRECTED GEODESICS CERTIFIED: {cert['DIRECTED_GEODESICS_CERTIFIED']}")
    print(f"READY FOR TRANSPORT HOLONOMY: {cert['READY_FOR_TRANSPORT_HOLONOMY']}")
    print(f"Certificate: {p}")


if __name__=="__main__":
    main()
