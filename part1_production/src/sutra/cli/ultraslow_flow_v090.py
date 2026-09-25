from __future__ import annotations

import argparse,json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from sutra.hierarchy.v071.block_preflight import (
    read_10x_cells_by_genes,normalized_expression_dense
)
from sutra.hierarchy.v071.resource_intake import sha256_file
from sutra.hierarchy.v074.effective_state import aggregate_supernode_expression
from sutra.cli.hierarchy_effective_flow_v0742 import current_relations,evaluate
from sutra.hierarchy.v076.geometry_hierarchy import (
    reconstruct_level0_mechanics,
    aggregate_mechanics,
    aggregate_covectors,
    full_effective_states,
    geometry_costs,
)
from sutra.hierarchy.v090.slow_flow import (
    SlowFlowConfig,choose_slow_batch,apply_contractions,
    summarize_numeric,shannon_from_sizes,flow_landmark_due
)
from sutra.hierarchy.v090.ledger import FlowLedger

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")


def require(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p


def build_cfg(raw):
    cfg=SlowFlowConfig(**raw)
    return cfg.validate()


def one(project_s,sample,cfg_dict):
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    t0=time.time()
    project=Path(project_s)
    cfg=build_cfg(cfg_dict)

    l0=project/"results"/"hierarchy_level0_v070"/sample
    v71=project/"results"/"hierarchy_v071_active_blocks"/sample
    v72=project/"results"/"hierarchy_v072_short_pilot"/sample
    v742=project/"results"/"hierarchy_v0742_effective_flow"/sample
    s1=project/"results"/"hierarchy_v075_local_geometry"/"step1_symmetric_base"/sample
    s3=project/"results"/"hierarchy_v075_local_geometry"/"step3_global_bound"/sample

    cells=pd.read_parquet(require(l0/"cells.parquet"))
    edge_rel=pd.read_parquet(require(v72/"level0_candidate_relations.parquet"))
    thresholds=json.loads(require(v72/"frozen_thresholds.json").read_text())
    frozen=json.loads(require(v742/"frozen_effective_thresholds.json").read_text())

    support_floor=float(frozen["communication_support"]["support_floor"])
    molecular_max=float(frozen["molecular_distance_max"])

    fmeta=json.loads(require(v71/"functional_resource_audit.json").read_text())
    lam=float(fmeta["lambda_g"])
    Lfunc=sparse.load_npz(require(v71/"functional_laplacian.npz")).toarray()

    manifest=json.loads(require(l0/"expression_manifest.json").read_text())
    matrix=Path(manifest["source_matrix"])
    if sha256_file(matrix)!=manifest["source_matrix_sha256"]:
        raise RuntimeError(f"{sample}: expression source hash changed")
    X,barcodes,genes=read_10x_cells_by_genes(matrix)
    Y=normalized_expression_dense(X)

    G=sparse.load_npz(require(s1/"symmetric_base_metric.npz")).toarray()
    Bcell=sparse.load_npz(require(s3/"directional_covectors_scaled.npz")).tocsr()

    mech,pressure_audit=reconstruct_level0_mechanics(edge_rel,len(cells))

    labels=np.arange(len(cells),dtype=np.int64)
    initial_n=len(cells)
    ledger=FlowLedger(
        project/"results"/"hierarchy_v090_ultraslow_flow"/"ledger",
        sample,
        archive_all_candidates=cfg.archive_all_candidates,
    )

    ledger.checkpoint_labels(0,labels)
    last_landmark=0.0
    total_merges=0
    exact_frontier_used=0
    rescue_modes=set()
    stop_reason=None

    for step in range(cfg.max_microsteps_safety):
        node_ids,mol_states,sizes=aggregate_supernode_expression(Y,labels)
        mech_states=aggregate_mechanics(mech,labels,node_ids)
        states=full_effective_states(mol_states,mech_states)
        Bnode=aggregate_covectors(Bcell,labels,node_ids)

        C0=current_relations(
            edge_rel,labels,node_ids,mol_states,Lfunc,lam,
            support_floor,1e-12
        )
        C=evaluate(C0,thresholds,molecular_max)
        C=geometry_costs(C,node_ids,states,Bnode,G)

        decision=choose_slow_batch(C,len(node_ids),step,cfg)
        elig=decision["eligible"]
        selected=decision["selected"]

        # Archive all candidate routes before modifying state.
        ledger.write_candidates(step,C)

        removed_fraction=(initial_n-len(node_ids))/initial_n
        row={
            "sample":sample,
            "microstep":int(step),
            "nodes_before":int(len(node_ids)),
            "level0_cells":int(initial_n),
            "cumulative_merges":int(total_merges),
            "removed_fraction_before":float(removed_fraction),
            "superedges":int(len(C)),
            "biologically_admissible":int(C.admissible.sum()) if len(C) else 0,
            "geometry_resolved_eligible":int(len(elig)),
            "selection_mode":decision["selection_mode"],
            "step_mode":decision["step_mode"],
            "batch_cap":int(decision["batch_cap"]),
            "batch_fraction":float(decision["batch_fraction"]),
            "routes_considered":decision["routes_considered"],
            "exact_frontier":bool(decision["exact_frontier"]),
            "selected_merges":int(len(selected)),
            "supernode_size_entropy":shannon_from_sizes(sizes),
            "supernode_size_max":int(np.max(sizes)) if len(sizes) else 0,
            "supernode_size_mean":float(np.mean(sizes)) if len(sizes) else 0.0,
        }

        # Track every cheap observable we already have on the current state.
        for col,prefix in [
            ("geometry_pair_cost","geom_cost"),
            ("alpha_cost","alpha_cost"),
            ("directional_correction","dir_corr"),
            ("geometry_reversal_ratio","reversal"),
            ("molecular_distance","molecular_dist"),
            ("mechanics_support_fraction","mech_support"),
            ("abs_tension_z","tension"),
            ("abs_delta_p_z","delta_p"),
            ("comm_support","comm_support"),
            ("comm_directionality","comm_direction"),
            ("comm_asymmetry","comm_asym"),
        ]:
            if col in C.columns:
                row.update(summarize_numeric(C[col],prefix))

        # Covector dual-norm distribution in the fixed G^{-1}.
        Ginv=np.linalg.inv(G)
        # sparse-row quadratic forms, matrix-free-ish for moderate 543 dim
        rho=[]
        for a in range(0,Bnode.shape[0],1024):
            z=min(Bnode.shape[0],a+1024)
            D=Bnode[a:z].toarray()
            rho.extend(np.sqrt(np.maximum(0,np.einsum("ij,jk,ik->i",D,Ginv,D))))
        row.update(summarize_numeric(rho,"directional_dual_norm"))

        if len(elig)==0:
            stop_reason="no_geometry_resolved_admissible_boundaries"
            row["stop_reason"]=stop_reason
            ledger.append_step(row)
            break

        if len(selected)==0:
            # This should be impossible because one eligible edge is itself a
            # disjoint matching. Treat as implementation failure, never as
            # scientific exhaustion.
            raise RuntimeError(
                f"{sample}: eligible boundaries remain but no contraction "
                "route was selected"
            )

        if decision["exact_frontier"]:
            exact_frontier_used+=1
        if decision["step_mode"]!="epsilon":
            rescue_modes.add(decision["step_mode"])

        selected=selected.copy()
        selected["survivor_node"]=np.minimum(
            selected.super_i.astype(np.int64),
            selected.super_j.astype(np.int64),
        )
        selected["removed_node"]=np.maximum(
            selected.super_i.astype(np.int64),
            selected.super_j.astype(np.int64),
        )
        ledger.write_merges(step,selected)

        labels=apply_contractions(labels,selected)
        total_merges+=len(selected)

        after_n=len(np.unique(labels))
        after_removed=(initial_n-after_n)/initial_n
        row["nodes_after"]=int(after_n)
        row["removed_fraction_after"]=float(after_removed)
        row["cumulative_merges_after"]=int(total_merges)

        due,new_landmark=flow_landmark_due(after_removed,last_landmark,cfg)
        if due:
            # Heavy observables are intentionally marked densely in reduction
            # space. The full state is exactly replayable from merge events +
            # checkpoints, so future heavy modules never require re-running the
            # contraction search.
            marker={
                "sample":sample,
                "microstep":int(step),
                "removed_fraction":float(after_removed),
                "landmark_fraction":float(new_landmark),
                "nodes":int(after_n),
                "replay_checkpoint_available":True,
                "requested_heavy_observables":[
                    "transport_component_structure",
                    "directed_geodesic_landmarks",
                    "transport_holonomy",
                    "holonomy_density",
                    "functional_pathway_state",
                    "mechanics_state",
                    "communication_state",
                ],
            }
            ledger.write_heavy_marker(step,marker)
            last_landmark=new_landmark

        if step%cfg.checkpoint_every==0:
            ledger.checkpoint_labels(step+1,labels)

        ledger.append_step(row)

    else:
        # A safety ceiling is never natural exhaustion. The engine must report
        # HOLD, and the configured rescue schedule should normally prevent it.
        stop_reason="microstep_safety_ceiling_with_unknown_remaining_frontier"

    ledger.checkpoint_labels(step+1,labels)

    final_nodes=len(np.unique(labels))
    natural=(stop_reason=="no_geometry_resolved_admissible_boundaries")
    summary={
        "sample":sample,
        "level0_cells":int(initial_n),
        "final_nodes":int(final_nodes),
        "total_merges":int(total_merges),
        "removed_fraction":float((initial_n-final_nodes)/initial_n),
        "microsteps":int(step+1),
        "stop_reason":stop_reason,
        "natural_exhaustion":bool(natural),
        "exact_frontier_steps":int(exact_frontier_used),
        "rescue_modes_used":sorted(rescue_modes),
        "pressure_audit_source":pressure_audit,
        "elapsed_seconds":float(time.time()-t0),
        "ledger_root":str(ledger.root),
        "status":"PASS" if natural else "HOLD",
    }
    (ledger.root/"flow_summary.json").write_text(
        json.dumps(summary,indent=2)+"\n"
    )
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument(
        "--config",
        default="configs/hierarchy_v090_ultraslow_flow.json"
    )
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    atlas=require(
        project/"results"/"hierarchy_v079_loop_normalized_geometry"/
        "loop_normalized_geometry_global_certificate.json"
    )
    d=json.loads(atlas.read_text())
    if d.get("READY_FOR_SPATIAL_GEOMETRY_STATISTICS") is not True:
        raise SystemExit("ERROR: v0.7.9 Level-0 geometry is not certified")

    raw=json.loads(require(project/a.config).read_text())
    cfg_dict=raw["slow_flow"]

    print("STRATA 0.9.0 | Ultraslow effective-state flow")
    print("Almost-continuous contraction schedule with full recomputation.")
    print("Scientific gates are never relaxed.")
    print("Multiple reduction routes are searched at every microstep.")
    print("Small frontiers are searched exactly before exhaustion.")
    print("Larger batches are runtime rescue only, never a scientific fallback.")
    print("Full streaming Flow Ledger enabled.\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(SAMPLES))) as ex:
        futs={
            ex.submit(one,str(project),s,cfg_dict):s
            for s in SAMPLES
        }
        for f in as_completed(futs):
            r=f.result();reports.append(r)
            print(
                f"[DONE] {r['sample']}: microsteps={r['microsteps']:,} "
                f"merges={r['total_merges']:,} "
                f"nodes={r['level0_cells']:,}->{r['final_nodes']:,} "
                f"removed={100*r['removed_fraction']:.2f}% "
                f"exact_frontier={r['exact_frontier_steps']} "
                f"rescue={','.join(r['rescue_modes_used']) or 'none'} "
                f"{r['stop_reason']} {r['status']}",
                flush=True
            )

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)

    out=project/"results"/"hierarchy_v090_ultraslow_flow"
    cert={
        "strata_version":"0.9.0",
        "stage":"ultraslow effective-state flow",
        "source_v079_certificate_sha256":sha256_file(atlas),
        "scientific_policy":{
            "almost_continuous_default":True,
            "full_effective_state_recomputed_every_microstep":True,
            "molecular_gate_relaxed":False,
            "mechanics_gate_relaxed":False,
            "communication_gate_relaxed":False,
            "geometry_resolution_relaxed":False,
            "multiple_reduction_routes_searched":True,
            "exact_matching_search_near_exhaustion":True,
            "larger_reduction_allowed_only_as_runtime_rescue":True,
            "natural_stop_requires_zero_eligible_boundaries":True,
            "safety_ceiling_counts_as_natural_stop":False,
            "full_candidate_archive_default":bool(
                cfg_dict.get("archive_all_candidates",True)
            ),
            "replayable_merge_lineage":True,
            "dense_heavy_observable_landmarks":True,
        },
        "sample_reports":reports,
        "ULTRASLOW_FLOW_GATE":"PASS" if gate else "HOLD",
        "FLOW_LEDGER_COMPLETE":bool(gate),
        "READY_FOR_EFFECTIVE_THEORY_OBSERVABLE_FLOW":bool(gate),
    }
    p=out/"ultraslow_flow_global_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print(f"\nULTRASLOW FLOW GATE: {cert['ULTRASLOW_FLOW_GATE']}")
    print(f"FLOW LEDGER COMPLETE: {cert['FLOW_LEDGER_COMPLETE']}")
    print(
        "READY FOR EFFECTIVE-THEORY OBSERVABLE FLOW: "
        f"{cert['READY_FOR_EFFECTIVE_THEORY_OBSERVABLE_FLOW']}"
    )
    print(f"Certificate: {p}")


if __name__=="__main__":
    main()
