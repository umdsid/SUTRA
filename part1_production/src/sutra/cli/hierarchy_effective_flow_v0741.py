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

from sutra.hierarchy.v071.block_preflight import (
    read_10x_cells_by_genes,
    normalized_expression_dense,
)
from sutra.hierarchy.v071.resource_intake import sha256_file
from sutra.hierarchy.v074.effective_state import (
    FlowConfig,
    freeze_communication_support_floor,
    communication_support_and_reciprocity,
    aggregate_supernode_expression,
    aggregate_supernode_centroids,
    aggregate_supernode_radii,
    functional_distance_between_states,
    current_boundary_relations,
    freeze_effective_molecular_threshold,
    freeze_supported_reciprocity_threshold,
    evaluate_effective_candidates,
    maximal_matching,
    contract,
    node_statistics,
    summarize_level,
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")


def require(p:Path)->Path:
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def writepq(df,path):
    pq.write_table(
        pa.Table.from_pandas(df,preserve_index=False),
        path,
        compression="zstd",
    )


def xy_from_cells(cells:pd.DataFrame):
    for a,b in [
        ("centroid_x","centroid_y"),
        ("x","y"),
        ("x_centroid","y_centroid"),
    ]:
        if a in cells.columns and b in cells.columns:
            return cells[[a,b]].to_numpy(np.float64)
    raise RuntimeError("cells table lacks recognized centroid columns")


def one(project_s:str,sample:str,cfg_dict:dict):
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )

    project=Path(project_s)
    cfg=FlowConfig(**cfg_dict)

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
    matrix_path=Path(manifest["source_matrix"])
    if sha256_file(matrix_path)!=manifest["source_matrix_sha256"]:
        raise RuntimeError(f"{sample}: expression source hash changed")

    X,barcodes,genes=read_10x_cells_by_genes(matrix_path)
    Y=normalized_expression_dense(X)
    xy=xy_from_cells(cells)

    # Reconstruct true Level-0 effective molecular distances from Level-0 cell states.
    labels0=np.arange(len(cells),dtype=np.int64)
    ids0,states0,sizes0=aggregate_supernode_expression(Y,labels0)
    B0=current_boundary_relations(
        edge_rel,labels0,ids0,states0,L,lam,
        communication_support_floor=cfg.eps,
        eps=cfg.eps,
    )

    comm_floor_meta=freeze_communication_support_floor(
        edge_rel,
        positive_quantile=cfg.communication_support_positive_quantile,
        eps=cfg.eps,
    )
    comm_floor=float(comm_floor_meta["communication_support_floor"])

    # Recompute Level-0 communication semantics with the real support floor.
    B0=current_boundary_relations(
        edge_rel,labels0,ids0,states0,L,lam,
        communication_support_floor=comm_floor,
        eps=cfg.eps,
    )

    molecular_distance_max=freeze_effective_molecular_threshold(B0,0.25)
    recip_meta=freeze_supported_reciprocity_threshold(
        edge_rel,comm_floor,q=0.50,eps=cfg.eps
    )
    comm_recip_min=float(recip_meta["comm_reciprocity_min"])

    out=project/"results"/"hierarchy_v0741_effective_flow"/sample
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)

    frozen={
        "molecular_distance_max":molecular_distance_max,
        "molecular_source_quantile":0.25,
        "communication_support":comm_floor_meta,
        "communication_reciprocity":recip_meta,
        "other_thresholds":{
            k:v for k,v in thresholds.items()
            if k not in ("comm_strength_min","comm_reciprocity_min")
        },
        "lambda_g":lam,
        "recomputed_at_later_levels":False,
    }
    (out/"frozen_effective_thresholds.json").write_text(
        json.dumps(frozen,indent=2)+"\n"
    )

    labels=np.arange(len(cells),dtype=np.int64)
    trajectory=[]
    merges=[]
    boundaries=[]
    level_stats=[]
    stop_reason=None

    for level in range(1,cfg.max_levels+1):
        before=labels.copy()
        node_ids,states,sizes=aggregate_supernode_expression(Y,labels)
        _,centroids=aggregate_supernode_centroids(xy,labels)
        radii=aggregate_supernode_radii(xy,labels,node_ids,centroids)

        B=current_boundary_relations(
            edge_rel,labels,node_ids,states,L,lam,
            communication_support_floor=comm_floor,
            eps=cfg.eps,
        )
        C=evaluate_effective_candidates(
            B,thresholds,molecular_distance_max,comm_recip_min
        )
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

        if stop_reason is not None:
            ns=node_statistics(labels,xy,node_ids,centroids,radii)
            lev=summarize_level(level,before,labels,C,S,ns)
            lev["stop_reason"]=stop_reason
            trajectory.append(lev)
            level_stats.append(pd.DataFrame([lev]))
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

        selected_keys=set(zip(
            selected.super_i.astype(int),
            selected.super_j.astype(int),
        ))
        C2=C.copy()
        C2["level"]=level
        C2["selected"]=[
            (int(a),int(b)) in selected_keys
            for a,b in zip(C2.super_i,C2.super_j)
        ]
        boundaries.append(C2)

        labels=contract(labels,S)

        node_ids2,states2,sizes2=aggregate_supernode_expression(Y,labels)
        _,centroids2=aggregate_supernode_centroids(xy,labels)
        radii2=aggregate_supernode_radii(xy,labels,node_ids2,centroids2)
        ns=node_statistics(labels,xy,node_ids2,centroids2,radii2)

        lev=summarize_level(level,before,labels,C,S,ns)
        trajectory.append(lev)
        level_stats.append(pd.DataFrame([lev]))

    else:
        stop_reason="max_level_safety_ceiling"

    final_nodes,final_states,final_sizes=aggregate_supernode_expression(Y,labels)
    _,final_centroids=aggregate_supernode_centroids(xy,labels)
    final_radii=aggregate_supernode_radii(
        xy,labels,final_nodes,final_centroids
    )
    final_node_stats=node_statistics(
        labels,xy,final_nodes,final_centroids,final_radii
    )

    writepq(pd.DataFrame({
        "cell_index":np.arange(len(labels),dtype=np.int64),
        "hierarchy_node":labels.astype(np.int64),
    }),out/"final_membership.parquet")
    writepq(final_node_stats,out/"final_node_statistics.parquet")

    M=pd.concat(merges,ignore_index=True) if merges else pd.DataFrame()
    BSTAT=pd.concat(boundaries,ignore_index=True) if boundaries else pd.DataFrame()
    LSTAT=pd.concat(level_stats,ignore_index=True) if level_stats else pd.DataFrame()

    writepq(M,out/"merge_trajectory.parquet")
    writepq(BSTAT,out/"boundary_statistics_all_levels.parquet")
    writepq(LSTAT,out/"level_statistics.parquet")

    if len(BSTAT):
        keep=[
            "level","admissible","selected",
            "pass_molecular","pass_mechanics_support",
            "pass_tension","pass_delta_p",
            "pass_communication_support",
            "pass_communication_reciprocity",
            "molecular_distance",
            "mechanics_support_fraction",
            "abs_tension_z","abs_delta_p_z",
            "comm_support","comm_supported",
            "comm_reciprocity","comm_asymmetry",
            "n_boundary_edges",
        ]
        writepq(BSTAT[keep],out/"figure_ready_boundary_trajectory.parquet")

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
        "final_nodes":int(len(final_nodes)),
        "fraction_nodes_removed":float(
            (len(cells)-len(final_nodes))/len(cells)
        ),
        "stop_reason":stop_reason,
        "natural_exhaustion":bool(natural),
        "communication_support_floor":comm_floor,
        "communication_positive_support_fraction":comm_floor_meta[
            "positive_support_fraction"
        ],
        "communication_reciprocity_threshold":comm_recip_min,
        "zero_support_is_reciprocal":False,
        "unsupported_reciprocity_is_na":True,
        "statistics_archive_complete":True,
    }
    report["status"]="PASS" if natural else "HOLD"

    (out/"effective_flow_summary.json").write_text(
        json.dumps(report,indent=2)+"\n"
    )
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--max-levels",type=int,default=250)
    ap.add_argument("--checkpoint-every",type=int,default=2)
    ap.add_argument(
        "--comm-support-positive-quantile",
        type=float,default=0.10
    )
    a=ap.parse_args()

    if not (0.0 <= a.comm_support_positive_quantile < 1.0):
        raise SystemExit("communication support quantile must be in [0,1)")

    project=Path(a.project_root).resolve()

    src=project/"results"/"hierarchy_v074_effective_flow"/"effective_flow_certificate.json"
    if not src.exists():
        raise SystemExit("ERROR: v0.7.4 certificate missing")
    d=json.loads(src.read_text())
    if d.get("EFFECTIVE_FLOW_GATE")!="PASS":
        raise SystemExit("ERROR: v0.7.4 gate is not PASS")

    cfg=FlowConfig(
        max_levels=a.max_levels,
        checkpoint_every=a.checkpoint_every,
        communication_support_positive_quantile=a.comm_support_positive_quantile,
    )

    print("STRATA 0.7.4.1 | Communication-support semantics correction")
    print("Zero communication is unsupported, never perfectly reciprocal.")
    print(
        "Support floor: frozen from positive Level-0 support "
        f"q={cfg.communication_support_positive_quantile:.2f}"
    )
    print("All other effective-state hierarchy rules remain unchanged.")
    print(f"Parallel specimens: {min(a.workers,len(SAMPLES))}\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(SAMPLES))) as ex:
        futs={
            ex.submit(one,str(project),s,cfg.__dict__.copy()):s
            for s in SAMPLES
        }
        for f in as_completed(futs):
            r=f.result();reports.append(r)
            print(
                f"[DONE] {r['sample']}: "
                f"levels={r['levels_executed']} "
                f"merges={r['total_contractions']:,} "
                f"nodes={r['n_level0_cells']:,}->{r['final_nodes']:,} "
                f"removed={100*r['fraction_nodes_removed']:.2f}% "
                f"comm_positive={100*r['communication_positive_support_fraction']:.1f}% "
                f"stop={r['stop_reason']} {r['status']}",
                flush=True,
            )

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)

    out=project/"results"/"hierarchy_v0741_effective_flow"
    out.mkdir(parents=True,exist_ok=True)
    cert={
        "strata_version":"0.7.4.1",
        "stage":"effective-state hierarchy with communication support semantics",
        "source_v074_certificate_sha256":sha256_file(src),
        "correction":{
            "zero_support_interpretation":"unsupported",
            "reciprocity_below_support_floor":"NA",
            "support_floor":"positive Level-0 support quantile",
            "support_quantile":cfg.communication_support_positive_quantile,
            "other_scientific_gates_changed":False,
        },
        "sample_reports":reports,
        "COMMUNICATION_SUPPORT_GATE":"PASS" if gate else "HOLD",
        "EFFECTIVE_FLOW_CORRECTED":"PASS" if gate else "HOLD",
        "READY_FOR_DIRECTIONAL_LOCAL_METRIC":bool(gate),
    }
    p=out/"effective_flow_v0741_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print(
        f"\nCOMMUNICATION SUPPORT GATE: "
        f"{cert['COMMUNICATION_SUPPORT_GATE']}"
    )
    print(
        "READY FOR DIRECTIONAL LOCAL METRIC: "
        f"{cert['READY_FOR_DIRECTIONAL_LOCAL_METRIC']}"
    )
    print(f"Certificate: {p}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
