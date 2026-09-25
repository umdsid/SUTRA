from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from strata_hierarchy.v071.block_preflight import (
    read_10x_cells_by_genes,
    normalized_expression_dense,
)
from strata_hierarchy.v071.resource_intake import sha256_file
from strata_hierarchy.v074.effective_state import (
    aggregate_supernode_expression,
    aggregate_supernode_centroids,
    aggregate_supernode_radii,
    functional_distance_between_states,
    supernode_boundary_table,
    freeze_effective_molecular_threshold,
    maximal_matching,
    contract,
    node_statistics,
)
from strata_hierarchy.v074.directional_comm import (
    DirectionalCommConfig,
    freeze_support_floor,
    support_floor_sensitivity,
    directional_state,
    block_gate_diagnostics,
    readiness_audit,
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")


def require(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p


def writepq(df,path):
    pq.write_table(
        pa.Table.from_pandas(df,preserve_index=False),
        path,
        compression="zstd",
    )


def xy_from_cells(cells):
    for a,b in [
        ("centroid_x","centroid_y"),
        ("x","y"),
        ("x_centroid","y_centroid"),
    ]:
        if a in cells.columns and b in cells.columns:
            return cells[[a,b]].to_numpy(np.float64)
    raise RuntimeError("cells table lacks centroid columns")


def current_relations(
    edge_rel,labels,node_ids,states,L,lam,support_floor,eps
):
    B=supernode_boundary_table(edge_rel,labels)
    if len(B)==0:
        return pd.DataFrame()

    idx={int(x):k for k,x in enumerate(node_ids)}
    rows=[]
    for (a,b),g in B.groupby(["super_i","super_j"],sort=False):
        ia=idx[int(a)];ib=idx[int(b)]
        dm=functional_distance_between_states(
            states[[ia]],states[[ib]],L,lam
        )[0]

        mech=(
            g.mechanics_tension_valid.astype(bool)
            & g.mechanics_delta_p_valid.astype(bool)
        )
        gm=g[mech]

        sij=float(np.nanmedian(g.comm_ij))
        sji=float(np.nanmedian(g.comm_ji))
        D=directional_state(
            np.array([sij]),np.array([sji]),support_floor,eps
        ).iloc[0]

        rows.append({
            "super_i":int(a),
            "super_j":int(b),
            "n_boundary_edges":int(len(g)),
            "molecular_distance":float(dm),
            "mechanics_support_fraction":float(mech.mean()),
            "abs_tension_z":(
                float(np.nanmedian(np.abs(gm.tension_z)))
                if len(gm) else np.nan
            ),
            "abs_delta_p_z":(
                float(np.nanmedian(np.abs(gm.delta_pressure_z)))
                if len(gm) else np.nan
            ),
            "comm_ij":sij,
            "comm_ji":sji,
            "comm_support":float(D.comm_support),
            "comm_supported":bool(D.comm_supported),
            "comm_directionality":float(D.comm_directionality)
                if np.isfinite(D.comm_directionality) else np.nan,
            "comm_asymmetry":float(D.comm_asymmetry)
                if np.isfinite(D.comm_asymmetry) else np.nan,
            "comm_reciprocity_descriptive":float(D.comm_reciprocity_descriptive)
                if np.isfinite(D.comm_reciprocity_descriptive) else np.nan,
        })
    return pd.DataFrame(rows)


def evaluate(B,thresholds,molecular_max):
    if len(B)==0: return B.copy()
    x=B.copy()
    x["pass_molecular"]=x.molecular_distance<=molecular_max
    x["pass_mechanics_support"]=(
        x.mechanics_support_fraction
        >= thresholds["min_mechanics_support_fraction"]
    )
    x["pass_tension"]=x.abs_tension_z<=thresholds["abs_tension_z_max"]
    x["pass_delta_p"]=x.abs_delta_p_z<=thresholds["abs_delta_p_z_max"]
    x["pass_communication_support"]=x.comm_supported.astype(bool)

    x["admissible"]=(
        x.pass_molecular
        & x.pass_mechanics_support
        & x.pass_tension
        & x.pass_delta_p
        & x.pass_communication_support
    )

    eps=1e-12
    parts=np.vstack([
        (molecular_max-x.molecular_distance)/(molecular_max+eps),
        (thresholds["abs_tension_z_max"]-x.abs_tension_z)
            /(thresholds["abs_tension_z_max"]+eps),
        (thresholds["abs_delta_p_z_max"]-x.abs_delta_p_z)
            /(thresholds["abs_delta_p_z_max"]+eps),
        x.mechanics_support_fraction
            - thresholds["min_mechanics_support_fraction"],
        np.where(x.comm_supported,np.log1p(x.comm_support),-1.0),
    ])
    x["ordering_merit"]=np.nanmean(parts,axis=0)
    return x


def level_summary(level,before,after,C,S):
    out={
        "level":int(level),
        "n_nodes_before":int(len(np.unique(before))),
        "n_nodes_after":int(len(np.unique(after))),
        "n_superedges":int(len(C)),
        "n_admissible":int(C.admissible.sum()) if len(C) else 0,
        "n_selected":int(S.selected.sum()) if len(S) else 0,
    }
    if len(C):
        out.update(block_gate_diagnostics(C))
        out.update({
            "comm_supported_fraction":float(C.comm_supported.mean()),
            "comm_support_median":float(np.nanmedian(C.comm_support)),
            "comm_abs_directionality_supported_median":float(
                np.nanmedian(C.loc[C.comm_supported,"comm_asymmetry"])
            ) if C.comm_supported.any() else np.nan,
            "comm_signed_directionality_supported_mean":float(
                np.nanmean(C.loc[C.comm_supported,"comm_directionality"])
            ) if C.comm_supported.any() else np.nan,
            "molecular_distance_median":float(np.nanmedian(C.molecular_distance)),
            "mechanics_support_median":float(
                np.nanmedian(C.mechanics_support_fraction)
            ),
        })
    return out


def one(project_s,sample,q,max_levels):
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    project=Path(project_s)

    l0=project/"results"/"hierarchy_level0_v070"/sample
    v71=project/"results"/"hierarchy_v071_active_blocks"/sample
    v72=project/"results"/"hierarchy_v072_short_pilot"/sample

    cells=pd.read_parquet(require(l0/"cells.parquet"))
    edge_rel=pd.read_parquet(require(v72/"level0_candidate_relations.parquet"))
    thresholds=json.loads(require(v72/"frozen_thresholds.json").read_text())
    fmeta=json.loads(require(v71/"functional_resource_audit.json").read_text())
    L=sparse.load_npz(require(v71/"functional_laplacian.npz")).toarray()
    lam=float(fmeta.get("lambda_g",1.0))

    manifest=json.loads(require(l0/"expression_manifest.json").read_text())
    matrix=Path(manifest["source_matrix"])
    if sha256_file(matrix)!=manifest["source_matrix_sha256"]:
        raise RuntimeError(f"{sample}: expression source hash changed")

    X,barcodes,genes=read_10x_cells_by_genes(matrix)
    Y=normalized_expression_dense(X)
    xy=xy_from_cells(cells)

    floor_meta=freeze_support_floor(edge_rel,q)
    floor=float(floor_meta["support_floor"])
    sens=support_floor_sensitivity(edge_rel,(0.01,0.05,0.10,0.20))

    labels=np.arange(len(cells),dtype=np.int64)
    ids0,S0,n0=aggregate_supernode_expression(Y,labels)
    B0=current_relations(edge_rel,labels,ids0,S0,L,lam,floor,1e-12)
    molecular_max=freeze_effective_molecular_threshold(B0,0.25)

    out=project/"results"/"hierarchy_v0742_effective_flow"/sample
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    writepq(sens,out/"communication_support_sensitivity.parquet")

    frozen={
        "molecular_distance_max":molecular_max,
        "molecular_source_quantile":0.25,
        "communication_support":floor_meta,
        "communication_support_sensitivity_quantiles":[0.01,0.05,0.10,0.20],
        "communication_directionality":"retained descriptive/geometric field; not an admissibility gate",
        "reciprocity":"descriptive only; not an admissibility gate",
        "other_thresholds":thresholds,
        "lambda_g":lam,
    }
    (out/"frozen_effective_thresholds.json").write_text(
        json.dumps(frozen,indent=2)+"\n"
    )

    merges=[];bounds=[];levels=[]
    stop_reason=None

    for level in range(1,max_levels+1):
        before=labels.copy()
        node_ids,states,sizes=aggregate_supernode_expression(Y,labels)
        B=current_relations(edge_rel,labels,node_ids,states,L,lam,floor,1e-12)
        C=evaluate(B,thresholds,molecular_max)
        S=maximal_matching(C)

        n_adm=int(C.admissible.sum()) if len(C) else 0
        n_sel=int(S.selected.sum()) if len(S) else 0

        if len(B)==0:
            stop_reason="no_remaining_supernode_boundaries"
        elif n_adm==0:
            stop_reason="no_admissible_boundaries"
        elif n_sel==0:
            stop_reason="no_disjoint_admissible_contractions"
        else:
            stop_reason=None

        if stop_reason:
            z=level_summary(level,before,labels,C,S)
            z["stop_reason"]=stop_reason
            levels.append(z)
            break

        selected=S[S.selected].copy()
        selected["level"]=level
        selected["survivor_node"]=np.minimum(
            selected.super_i.astype(np.int64),
            selected.super_j.astype(np.int64)
        )
        selected["removed_node"]=np.maximum(
            selected.super_i.astype(np.int64),
            selected.super_j.astype(np.int64)
        )
        merges.append(selected)

        keys=set(zip(selected.super_i.astype(int),selected.super_j.astype(int)))
        C2=C.copy()
        C2["level"]=level
        C2["selected"]=[
            (int(a),int(b)) in keys
            for a,b in zip(C2.super_i,C2.super_j)
        ]
        bounds.append(C2)

        labels=contract(labels,S)
        levels.append(level_summary(level,before,labels,C,S))

    else:
        stop_reason="max_level_safety_ceiling"

    M=pd.concat(merges,ignore_index=True) if merges else pd.DataFrame()
    BSTAT=pd.concat(bounds,ignore_index=True) if bounds else pd.DataFrame()
    LSTAT=pd.DataFrame(levels)

    writepq(M,out/"merge_trajectory.parquet")
    writepq(BSTAT,out/"boundary_statistics_all_levels.parquet")
    writepq(LSTAT,out/"level_statistics.parquet")
    writepq(pd.DataFrame({
        "cell_index":np.arange(len(labels),dtype=np.int64),
        "hierarchy_node":labels.astype(np.int64),
    }),out/"final_membership.parquet")

    D0=directional_state(edge_rel.comm_ij,edge_rel.comm_ji,floor)
    ready=readiness_audit(
        D0,sens,
        M if len(M) else None
    )

    # Prospective traps for v0.7.5.
    preflight={
        **ready,
        "support_floor_quantile":q,
        "support_floor":floor,
        "support_positive_fraction":floor_meta["positive_fraction"],
        "molecular_distance_threshold_positive":bool(molecular_max>0),
        "all_finite_supported_directionality":bool(
            ready["finite_directionality_fraction"]>=0.999999
        ),
        "signed_directionality_available_for_next_metric":bool(
            ready["n_supported"]>0
        ),
    }
    (out/"directional_metric_readiness.json").write_text(
        json.dumps(preflight,indent=2)+"\n"
    )

    natural=stop_reason in {
        "no_remaining_supernode_boundaries",
        "no_admissible_boundaries",
        "no_disjoint_admissible_contractions",
    }

    report={
        "sample":sample,
        "n_level0_cells":int(len(cells)),
        "levels_executed":int(LSTAT.level.max()) if len(LSTAT) else 0,
        "total_contractions":int(len(M)),
        "final_nodes":int(len(np.unique(labels))),
        "fraction_nodes_removed":float(
            (len(cells)-len(np.unique(labels)))/len(cells)
        ),
        "stop_reason":stop_reason,
        "natural_exhaustion":bool(natural),
        "communication_support_floor":floor,
        "communication_positive_fraction":floor_meta["positive_fraction"],
        "directionality_used_as_gate":False,
        "reciprocity_used_as_gate":False,
        "metric_readiness_structural":bool(preflight["metric_readiness_structural"]),
        "support_floor_sensitivity_span":preflight["support_floor_sensitivity_span"],
    }
    report["status"]="PASS" if (
        natural and preflight["metric_readiness_structural"]
    ) else "HOLD"
    (out/"effective_flow_summary.json").write_text(
        json.dumps(report,indent=2)+"\n"
    )
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--max-levels",type=int,default=250)
    ap.add_argument("--comm-support-positive-quantile",type=float,default=0.10)
    a=ap.parse_args()

    project=Path(a.project_root).resolve()
    src=require(
        project/"results"/"hierarchy_v0741_effective_flow"/
        "effective_flow_v0741_certificate.json"
    )
    d=json.loads(src.read_text())
    if d.get("READY_FOR_DIRECTIONAL_LOCAL_METRIC") is not True:
        raise SystemExit("ERROR: v0.7.4.1 did not pass readiness gate")

    print("STRATA 0.7.4.2 | Supported directional communication")
    print("Communication support remains an admissibility requirement.")
    print("Directionality/reciprocity are retained as geometry/statistics, not merger vetoes.")
    print("Adversarial readiness audit for the next local metric is enabled.")
    print(f"Parallel specimens: {min(a.workers,len(SAMPLES))}\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(SAMPLES))) as ex:
        futs={
            ex.submit(
                one,str(project),s,
                a.comm_support_positive_quantile,
                a.max_levels
            ):s for s in SAMPLES
        }
        for f in as_completed(futs):
            r=f.result();reports.append(r)
            print(
                f"[DONE] {r['sample']}: "
                f"levels={r['levels_executed']} "
                f"merges={r['total_contractions']:,} "
                f"removed={100*r['fraction_nodes_removed']:.2f}% "
                f"comm_positive={100*r['communication_positive_fraction']:.1f}% "
                f"metric_ready={r['metric_readiness_structural']} "
                f"{r['status']}",
                flush=True,
            )

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)

    out=project/"results"/"hierarchy_v0742_effective_flow"
    out.mkdir(parents=True,exist_ok=True)
    cert={
        "strata_version":"0.7.4.2",
        "stage":"supported directional communication effective flow",
        "source_v0741_certificate_sha256":sha256_file(src),
        "correction":{
            "communication_support_is_admissibility":True,
            "directionality_is_admissibility":False,
            "reciprocity_is_admissibility":False,
            "signed_directionality_retained":True,
            "zero_support_directionality":"NA",
        },
        "adversarial_preflight":{
            "support_floor_sensitivity":True,
            "bounded_directionality_check":True,
            "finite_supported_directionality_check":True,
            "selected_merge_support_check":True,
            "block_gate_degeneracy_diagnostics":True,
        },
        "sample_reports":reports,
        "SUPPORTED_DIRECTIONAL_COMM_GATE":"PASS" if gate else "HOLD",
        "READY_FOR_DIRECTIONAL_LOCAL_METRIC":bool(gate),
    }
    p=out/"effective_flow_v0742_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print(f"\nSUPPORTED DIRECTIONAL COMM GATE: {cert['SUPPORTED_DIRECTIONAL_COMM_GATE']}")
    print(f"READY FOR DIRECTIONAL LOCAL METRIC: {cert['READY_FOR_DIRECTIONAL_LOCAL_METRIC']}")
    print(f"Certificate: {p}")


if __name__=="__main__":
    main()
