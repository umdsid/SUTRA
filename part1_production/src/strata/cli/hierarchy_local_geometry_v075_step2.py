from __future__ import annotations
import argparse,json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from strata_hierarchy.v071.block_preflight import (
    read_10x_cells_by_genes,normalized_expression_dense
)
from strata_hierarchy.v071.resource_intake import sha256_file
from strata_hierarchy.v075.metric_core import (
    canonical_feature_order,feature_order_sha256
)
from strata_hierarchy.v075.directional_covector import (
    build_raw_node_covectors,dual_norms_spd,sparse_sha256,summarize_covectors
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def require(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p

def writepq(df,path):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),path,compression="zstd")

def registry_path(project,sample):
    p=project/"results"/"hierarchy_v071_active_blocks"/sample/"canonical_signaling_registry.parquet"
    if p.exists(): return p
    p=project/"resources"/"strata_hierarchy_v0711"/"signaling"/"cellchat_typed_interactions.csv"
    if p.exists(): return p
    raise FileNotFoundError("canonical signaling registry")

def readreg(p):
    return pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p,low_memory=False)

def one(project,sample):
    l0=project/"results"/"hierarchy_level0_v070"/sample
    v72=project/"results"/"hierarchy_v072_short_pilot"/sample
    s1=project/"results"/"hierarchy_v075_local_geometry"/"step1_symmetric_base"/sample
    v742=project/"results"/"hierarchy_v0742_effective_flow"/sample

    G=sparse.load_npz(require(s1/"symmetric_base_metric.npz")).toarray()
    s1cert=json.loads(require(s1/"symmetric_base_certificate.json").read_text())

    edge_rel=pd.read_parquet(require(v72/"level0_candidate_relations.parquet"))
    frozen=json.loads(require(v742/"frozen_effective_thresholds.json").read_text())
    floor=float(frozen["communication_support"]["support_floor"])

    manifest=json.loads(require(l0/"expression_manifest.json").read_text())
    matrix=Path(manifest["source_matrix"])
    if sha256_file(matrix)!=manifest["source_matrix_sha256"]:
        raise RuntimeError(f"{sample}: expression source hash changed")
    X,barcodes,genes=read_10x_cells_by_genes(matrix)
    Y=normalized_expression_dense(X)
    genes=tuple(map(str,genes))

    features=canonical_feature_order(genes)
    if feature_order_sha256(features)!=s1cert["state_space"]["feature_order_sha256"]:
        raise RuntimeError(f"{sample}: Step-1 feature order mismatch")

    rp=registry_path(project,sample)
    reg=readreg(rp)

    B,stats,channels=build_raw_node_covectors(
        Y,edge_rel,genes,reg,floor
    )
    if B.shape!=G.shape:
        # G shape is dxd; B is nxd
        if B.shape[1]!=G.shape[0]:
            raise RuntimeError(
                f"{sample}: covector dimension {B.shape[1]} != metric dimension {G.shape[0]}"
            )

    dual=dual_norms_spd(G,B)
    stats["raw_covector_dual_norm"]=dual
    summary=summarize_covectors(stats,dual)

    out=project/"results"/"hierarchy_v075_local_geometry"/"step2_directional_covector"/sample
    out.mkdir(parents=True,exist_ok=True)
    sparse.save_npz(out/"raw_directional_covectors.npz",B,compressed=True)
    writepq(stats,out/"raw_directional_covector_statistics.parquet")
    writepq(channels,out/"directional_channel_contributions.parquet")

    report={
        "sample":sample,
        "support_floor":floor,
        "state_dimension":int(B.shape[1]),
        "n_cells":int(B.shape[0]),
        "feature_order_sha256":feature_order_sha256(features),
        "registry_path":str(rp),
        "registry_sha256":sha256_file(rp),
        "n_panel_supported_channels":int(len(channels)),
        "raw_covector_sha256":sparse_sha256(B),
        "normalization":"divide node channel imbalance by total supported incident channel flux",
        "directional_coordinates":"molecular axes only in Step 2",
        "scaling_for_finsler_bound_applied":False,
        "summary":summary,
        "status":"PASS" if (
            summary["all_dual_norms_finite"]
            and summary["n_nodes_with_supported_incident_flux"]>0
        ) else "HOLD",
    }
    (out/"directional_covector_certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    source=require(
        project/"results"/"hierarchy_v075_local_geometry"/"step1_symmetric_base"/
        "symmetric_base_global_certificate.json"
    )
    d=json.loads(source.read_text())
    if d.get("READY_FOR_DIRECTIONAL_COVECTOR") is not True:
        raise SystemExit("ERROR: Step 1 did not permit directional covector construction")

    print("STRATA 0.7.5 Step 2 | Raw local directional covector")
    print("Building node-local channel-resolved communication covectors.")
    print("No Finsler-bound scaling is applied yet.\n")

    reports=[]
    for s in SAMPLES:
        r=one(project,s); reports.append(r)
        q=r["summary"]
        print(
            f"[DONE] {s}: channels={r['n_panel_supported_channels']} "
            f"supported_nodes={100*q['supported_incident_node_fraction']:.1f}% "
            f"nonzero_b={100*q['nonzero_raw_covector_fraction']:.1f}% "
            f"dual_q95={q['dual_norm_supported_q95']:.4g} "
            f"dual_max={q['dual_norm_max']:.4g} {r['status']}"
        )

    gate=all(r["status"]=="PASS" for r in reports)
    maxima=[r["summary"]["dual_norm_max"] for r in reports]
    out=project/"results"/"hierarchy_v075_local_geometry"/"step2_directional_covector"
    cert={
        "strata_version":"0.7.5-step2",
        "stage":"raw local directional covector",
        "source_step1_certificate_sha256":sha256_file(source),
        "construction":{
            "node_local":True,
            "channel_resolved":True,
            "measured_contact_support_only":True,
            "unsupported_edges_excluded":True,
            "edge_storage_orientation_invariant_by_construction":True,
            "mechanics_directional_components":"zero",
            "finsler_bound_scaling_applied":False,
        },
        "sample_reports":reports,
        "global_raw_dual_norm_max":float(max(maxima)),
        "DIRECTIONAL_COVECTOR_GATE":"PASS" if gate else "HOLD",
        "READY_FOR_GLOBAL_BOUND_CALIBRATION":bool(gate),
    }
    p=out/"directional_covector_global_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")
    print(f"\nDIRECTIONAL COVECTOR GATE: {cert['DIRECTIONAL_COVECTOR_GATE']}")
    print(f"READY FOR GLOBAL BOUND CALIBRATION: {cert['READY_FOR_GLOBAL_BOUND_CALIBRATION']}")
    print(f"GLOBAL RAW DUAL NORM MAX: {cert['global_raw_dual_norm_max']:.6g}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
