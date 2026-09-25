from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from sutra.hierarchy.v071.block_preflight import (
    sha256_file,
    read_10x_cells_by_genes,
    normalized_expression_dense,
    discover_functional_gmts,
    functional_prior,
    contact_edge_table,
    molecular_edge_distance,
    robust_location_scale,
    robust_standardize,
    mechanical_relations,
    audit_signaling_resources,
    molecular_prior_status,
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")


def require(p:Path)->Path:
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_freeze(project:Path):
    p=require(project/"results"/"hierarchy_level0_v070"/"level0_freeze_certificate.json")
    d=json.loads(p.read_text())
    if d.get("LEVEL0_FREEZE")!="PASS" or not d.get("HIERARCHY_INPUT_READY",False):
        raise RuntimeError("v0.7.0 Level-0 freeze is not hierarchy-ready")
    return p,d


def one(project_s:str,sample:str,lam:float,batch_size:int):
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    project=Path(project_s)
    l0=project/"results"/"hierarchy_level0_v070"/sample
    cells=pd.read_parquet(require(l0/"cells.parquet"))
    interfaces=pd.read_parquet(require(l0/"interfaces.parquet"))
    features=pd.read_parquet(require(l0/"features.parquet"))
    manifest=json.loads(require(l0/"expression_manifest.json").read_text())

    matrix_path=Path(manifest["source_matrix"])
    if sha256_file(matrix_path)!=manifest["source_matrix_sha256"]:
        raise RuntimeError(f"{sample}: expression source hash changed")

    X,barcodes,genes=read_10x_cells_by_genes(matrix_path)
    if X.shape!=(len(cells),len(features)):
        raise RuntimeError(f"{sample}: Level-0/expression dimension mismatch")
    if tuple(map(str,barcodes))!=tuple(cells.cell_id.astype(str)):
        raise RuntimeError(f"{sample}: barcode order mismatch")
    if tuple(map(str,genes))!=tuple(features.feature_name.astype(str)):
        # feature IDs/names can differ by export convention.  Require one of
        # Level-0 name or ID to match H5 order exactly.
        if "feature_id" not in features or tuple(map(str,genes))!=tuple(features.feature_id.astype(str)):
            raise RuntimeError(f"{sample}: feature order mismatch")

    resources=project/"resources"
    gmts=discover_functional_gmts(resources)
    W,L,faudit=functional_prior(tuple(map(str,genes)),gmts)
    fstatus=molecular_prior_status(faudit)

    Y=normalized_expression_dense(X)
    E=contact_edge_table(interfaces)
    dmol=molecular_edge_distance(Y,E,L,lam=lam,batch_size=batch_size)
    smol=robust_location_scale(dmol)
    zmol=robust_standardize(dmol,smol)

    stau,sdp,ztau,zdp=mechanical_relations(E)

    comm_audit,registry=audit_signaling_resources(
        resources,set(map(str,genes))
    )

    R=E[["interface_id","cell_i_index","cell_j_index","owner_patch_id"]].copy()
    R["molecular_distance"]=dmol
    R["molecular_z"]=zmol
    R["tension"]=E.tension.to_numpy(np.float64)
    R["tension_z"]=ztau
    R["delta_pressure"]=E.delta_pressure.to_numpy(np.float64)
    R["delta_pressure_z"]=zdp
    R["mechanics_tension_valid"]=np.isfinite(R.tension)
    R["mechanics_delta_p_valid"]=np.isfinite(R.delta_pressure)
    R["mechanics_complete"]=(
        R.mechanics_tension_valid & R.mechanics_delta_p_valid
    )

    out=project/"results"/"hierarchy_v071_active_blocks"/sample
    out.mkdir(parents=True,exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(R,preserve_index=False),
        out/"contact_block_relations.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pandas(registry,preserve_index=False),
        out/"canonical_signaling_registry.parquet",
        compression="zstd",
    )

    # Save compact functional matrix rather than dense JSON.
    sparse.save_npz(out/"functional_prior_W.npz",sparse.csr_matrix(W))
    sparse.save_npz(out/"functional_laplacian.npz",sparse.csr_matrix(L))

    scales={
        "molecular_distance":smol,
        "tension":stau,
        "delta_pressure":sdp,
        "rule":"(z - median(Level0 valid support edges)) / (IQR + eps)",
        "frozen_for_trajectory":True,
        "recompute_at_later_levels":False,
    }
    (out/"block_scales.json").write_text(json.dumps(scales,indent=2)+"\n")
    (out/"functional_resource_audit.json").write_text(
        json.dumps({
            "status":fstatus,
            "lambda_g":lam,
            "resources":faudit,
        },indent=2)+"\n"
    )
    (out/"communication_resource_audit.json").write_text(
        json.dumps(comm_audit,indent=2)+"\n"
    )

    report={
        "sample":sample,
        "n_cells":int(len(cells)),
        "n_genes":int(len(genes)),
        "n_contact_edges":int(len(E)),
        "functional_prior_status":fstatus["status"],
        "functional_layers_used":fstatus["n_resource_layers_used"],
        "molecular_scale_valid":bool(smol["scale_valid"]),
        "mechanical_tension_scale_valid":bool(stau["scale_valid"]),
        "mechanical_delta_p_scale_valid":bool(sdp["scale_valid"]),
        "mechanics_complete_edges":int(R.mechanics_complete.sum()),
        "mechanics_complete_fraction":float(R.mechanics_complete.mean()) if len(R) else 0.0,
        "communication_status":comm_audit["status"],
        "communication_panel_supported_interactions":comm_audit["n_panel_supported_interactions"],
        "hierarchy_reduction_performed":False,
    }
    # Gate D cannot pass without all three active blocks ready.
    report["status"]="PASS" if (
        report["functional_prior_status"]=="PASS"
        and report["molecular_scale_valid"]
        and report["mechanical_tension_scale_valid"]
        and report["mechanical_delta_p_scale_valid"]
        and report["communication_status"]=="PASS"
    ) else "HOLD"

    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--lambda-g",type=float,default=1.0)
    ap.add_argument("--batch-size",type=int,default=4096)
    a=ap.parse_args()
    if a.lambda_g<0:
        raise SystemExit("--lambda-g must be nonnegative")

    project=Path(a.project_root).resolve()
    freeze_path,_=load_freeze(project)

    print("STRATA 0.7.1 | Active-block readiness + Level-0 normalization")
    print("No hierarchy reduction is performed.")
    print("Active blocks: functional molecular state | primitive mechanics | directed communication")
    print(f"Parallel specimen workers: {min(a.workers,len(SAMPLES))}\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(SAMPLES))) as ex:
        futs={
            ex.submit(one,str(project),s,a.lambda_g,a.batch_size):s
            for s in SAMPLES
        }
        for f in as_completed(futs):
            r=f.result();reports.append(r)
            print(
                f"[DONE] {r['sample']}: "
                f"edges={r['n_contact_edges']:,} "
                f"func={r['functional_prior_status']} "
                f"mech_complete={100*r['mechanics_complete_fraction']:.1f}% "
                f"comm={r['communication_status']} "
                f"{r['status']}",
                flush=True,
            )

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    all_ready=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v071_active_blocks"
    out.mkdir(parents=True,exist_ok=True)
    cert={
        "strata_version":"0.7.1",
        "stage":"active-block readiness and Level-0 robust normalization",
        "source_level0_freeze_sha256":sha256_file(freeze_path),
        "source_level0_freeze":"PASS",
        "hierarchy_reduction_performed":False,
        "block_rule":{
            "active":[
                "functional molecular state",
                "primitive mechanics",
                "directed communication",
            ],
            "support":["measured contact topology"],
            "diagnostic_only":[
                "morphology",
                "stress and descendants",
                "spatial-gene cross association",
            ],
            "missing_mechanics":"missing/invalid, never zero",
        },
        "normalization":{
            "robust_level0_rule":"(z - median) / (IQR + eps)",
            "scales_frozen_for_trajectory":True,
            "scales_recomputed_at_later_levels":False,
            "lambda_g":a.lambda_g,
        },
        "sample_reports":reports,
        "ACTIVE_BLOCK_GATE":"PASS" if all_ready else "HOLD",
        "SHORT_HIERARCHY_READY":bool(all_ready),
    }
    p=out/"active_block_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print(f"\nACTIVE BLOCK GATE: {cert['ACTIVE_BLOCK_GATE']}")
    print(f"SHORT HIERARCHY READY: {cert['SHORT_HIERARCHY_READY']}")
    print(f"Certificate: {p}")
    if not all_ready:
        print("\nHOLD is diagnostic: inspect each sample's functional_resource_audit.json")
        print("and communication_resource_audit.json. No fallback resource inference is performed.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
