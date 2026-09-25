from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import sparse

from sutra.hierarchy.v071.block_preflight import read_10x_cells_by_genes
from sutra.hierarchy.v071.resource_intake import sha256_file
from sutra.hierarchy.v075.metric_core import (
    SymmetricBaseConfig,canonical_feature_order,feature_order_sha256,
    build_symmetric_base_metric,certify_symmetric_base_metric,sparse_matrix_sha256
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def require(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p

def get_lambda_g(audit):
    if "lambda_g" in audit: return float(audit["lambda_g"])
    if isinstance(audit.get("metric"),dict) and "lambda_g" in audit["metric"]:
        return float(audit["metric"]["lambda_g"])
    raise RuntimeError("functional resource audit does not contain lambda_g")

def one(project,sample,cfg):
    l0=project/"results"/"hierarchy_level0_v070"/sample
    v71=project/"results"/"hierarchy_v071_active_blocks"/sample
    audit=json.loads(require(v71/"functional_resource_audit.json").read_text())
    lam=get_lambda_g(audit)
    L_path=require(v71/"functional_laplacian.npz")
    L=sparse.load_npz(L_path).tocsr()

    manifest=json.loads(require(l0/"expression_manifest.json").read_text())
    matrix=Path(manifest["source_matrix"])
    if sha256_file(matrix)!=manifest["source_matrix_sha256"]:
        raise RuntimeError(f"{sample}: expression source hash changed")
    _,_,genes=read_10x_cells_by_genes(matrix)
    genes=tuple(map(str,genes))
    if L.shape!=(len(genes),len(genes)):
        raise RuntimeError(f"{sample}: Laplacian {L.shape} != {len(genes)} genes")

    thr=json.loads(require(
        project/"results"/"hierarchy_v072_short_pilot"/sample/"frozen_thresholds.json"
    ).read_text())
    for k in ("abs_tension_z_max","abs_delta_p_z_max"):
        if k not in thr or not np.isfinite(float(thr[k])):
            raise RuntimeError(f"{sample}: mechanics threshold {k} missing")

    features=canonical_feature_order(genes)
    G=build_symmetric_base_metric(L,lam,len(genes),cfg)
    cert=certify_symmetric_base_metric(G,cfg)

    out=project/"results"/"hierarchy_v075_local_geometry"/"step1_symmetric_base"/sample
    out.mkdir(parents=True,exist_ok=True)
    sparse.save_npz(out/"symmetric_base_metric.npz",G,compressed=True)
    report={
        "sample":sample,
        "state_space":{
            "n_measured_genes":len(genes),
            "mechanics_coordinates":["tension_z","delta_pressure_z"],
            "dimension":len(features),
            "feature_order_sha256":feature_order_sha256(features),
        },
        "functional":{
            "lambda_g":lam,
            "functional_laplacian_path":str(L_path),
            "functional_laplacian_sha256":sha256_file(L_path),
        },
        "mechanics":{
            "tension_weight":cfg.mechanics_tension_weight,
            "delta_p_weight":cfg.mechanics_delta_p_weight,
            "coordinates_are_upstream_standardized":True,
            "tension_gate":float(thr["abs_tension_z_max"]),
            "delta_p_gate":float(thr["abs_delta_p_z_max"]),
        },
        "metric":{
            **cert,
            "metric_sha256":sparse_matrix_sha256(G),
            "directional_term_present":False,
            "construction":"blockdiag(I + lambda_g L_func, w_tau, w_dp)",
        },
        "status":"PASS" if cert["certificate_pass"] else "HOLD",
    }
    (out/"symmetric_base_certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--mechanics-tension-weight",type=float,default=1.0)
    ap.add_argument("--mechanics-delta-p-weight",type=float,default=1.0)
    ap.add_argument("--max-condition-number",type=float,default=1e8)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    source=require(project/"results"/"hierarchy_v0743_channel_direction_audit"/"channel_direction_audit_certificate.json")
    d=json.loads(source.read_text())
    if d.get("DIRECTIONAL_REPRESENTATION_READY") is not True:
        raise SystemExit("ERROR: v0.7.4.3 directional representation not ready")

    cfg=SymmetricBaseConfig(
        mechanics_tension_weight=a.mechanics_tension_weight,
        mechanics_delta_p_weight=a.mechanics_delta_p_weight,
        max_condition_number=a.max_condition_number,
    )
    print("STRATA 0.7.5 Step 1 | Symmetric local-geometry base")
    print("Constructing alpha(v)=sqrt(v^T G v).")
    print("No directional communication term is introduced yet.\n")
    reports=[]
    for sample in SAMPLES:
        r=one(project,sample,cfg); reports.append(r)
        m=r["metric"]
        print(f"[DONE] {sample}: dim={m['dimension']} lambda_min={m['lambda_min']:.6g} lambda_max={m['lambda_max']:.6g} cond={m['condition_number']:.6g} {r['status']}")
    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v075_local_geometry"/"step1_symmetric_base"
    cert={
        "strata_version":"0.7.5-step1",
        "stage":"symmetric local-geometry base",
        "source_v0743_certificate_sha256":sha256_file(source),
        "construction":{
            "molecular_block":"I + lambda_g L_func",
            "mechanics_axes":["tension_z","delta_pressure_z"],
            "mechanics_weights":[cfg.mechanics_tension_weight,cfg.mechanics_delta_p_weight],
            "directional_term_present":False,
            "normalization_recomputed":False,
        },
        "sample_reports":reports,
        "SYMMETRIC_BASE_GATE":"PASS" if gate else "HOLD",
        "READY_FOR_DIRECTIONAL_COVECTOR":bool(gate),
    }
    p=out/"symmetric_base_global_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")
    print(f"\nSYMMETRIC BASE GATE: {cert['SYMMETRIC_BASE_GATE']}")
    print(f"READY FOR DIRECTIONAL COVECTOR: {cert['READY_FOR_DIRECTIONAL_COVECTOR']}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
