from __future__ import annotations

import argparse,json,os,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from sutra.hierarchy.v071.block_preflight import (
    read_10x_cells_by_genes,normalized_expression_dense
)
from sutra.hierarchy.v071.resource_intake import sha256_file
from sutra.hierarchy.v074.effective_state import (
    aggregate_supernode_expression,
)
from sutra.cli.hierarchy_effective_flow_v0742 import (
    current_relations,evaluate,
)
from sutra.hierarchy.v076.geometry_hierarchy import (
    reconstruct_level0_mechanics,
    aggregate_mechanics,
    aggregate_covectors,
    full_effective_states,
    geometry_costs,
    geometry_matching,
    baseline_matching,
    matching_overlap,
    contract_labels,
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


def one(project_s,sample,max_levels):
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
    L=sparse.load_npz(require(v71/"functional_laplacian.npz")).toarray()

    manifest=json.loads(require(l0/"expression_manifest.json").read_text())
    matrix=Path(manifest["source_matrix"])
    if sha256_file(matrix)!=manifest["source_matrix_sha256"]:
        raise RuntimeError(f"{sample}: expression source hash changed")
    X,barcodes,genes=read_10x_cells_by_genes(matrix)
    Y=normalized_expression_dense(X)

    G=sparse.load_npz(require(s1/"symmetric_base_metric.npz")).toarray()
    Bcell=sparse.load_npz(require(s3/"directional_covectors_scaled.npz")).tocsr()

    if G.shape[0] != Y.shape[1]+2:
        raise RuntimeError(f"{sample}: metric/state dimension mismatch")
    if Bcell.shape != (len(cells),G.shape[0]):
        raise RuntimeError(f"{sample}: cell covector shape mismatch")

    mech,pressure_audit=reconstruct_level0_mechanics(edge_rel,len(cells))

    out=project/"results"/"hierarchy_v076_geometry_aware"/sample
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    writepq(mech,out/"level0_mechanical_state.parquet")
    (out/"pressure_reconstruction_audit.json").write_text(
        json.dumps(pressure_audit,indent=2)+"\n"
    )

    labels=np.arange(len(cells),dtype=np.int64)
    level_rows=[]
    boundary_parts=[]
    merge_parts=[]
    stop_reason=None

    for level in range(1,max_levels+1):
        before=labels.copy()
        node_ids,mol_states,sizes=aggregate_supernode_expression(Y,labels)
        mech_states=aggregate_mechanics(mech,labels,node_ids)
        states=full_effective_states(mol_states,mech_states)
        Bnode=aggregate_covectors(Bcell,labels,node_ids)

        # Convex averaging of cell covectors cannot violate the Step-3 strict bound,
        # but we still archive the Euclidean norm and defer exact dual checks to stats.
        C0=current_relations(
            edge_rel,labels,node_ids,mol_states,L,lam,support_floor,1e-12
        )
        C=evaluate(C0,thresholds,molecular_max)
        Cg=geometry_costs(C,node_ids,states,Bnode,G)

        # Geometric state resolution is not a replacement biological gate.
        # It is required only because unresolved state has no defined cost.
        n_adm=int(Cg.admissible.sum()) if len(Cg) else 0
        n_geom_eligible=int(
            (
                Cg.admissible.astype(bool)
                & Cg.geometry_state_resolved.astype(bool)
                & np.isfinite(Cg.geometry_pair_cost)
            ).sum()
        ) if len(Cg) else 0

        GM=geometry_matching(Cg)
        BM=baseline_matching(Cg)
        overlap=matching_overlap(GM,BM)

        n_sel=int(GM.selected_geometry.sum()) if len(GM) else 0

        if len(Cg)==0:
            stop_reason="no_remaining_supernode_boundaries"
        elif n_adm==0:
            stop_reason="no_biologically_admissible_boundaries"
        elif n_geom_eligible==0:
            stop_reason="no_geometry_resolved_admissible_boundaries"
        elif n_sel==0:
            stop_reason="no_disjoint_geometry_resolved_contractions"
        else:
            stop_reason=None

        row={
            "level":level,
            "n_nodes_before":int(len(node_ids)),
            "n_superedges":int(len(Cg)),
            "n_biologically_admissible":n_adm,
            "n_geometry_resolved_admissible":n_geom_eligible,
            "geometry_resolution_fraction_of_admissible":
                float(n_geom_eligible/n_adm) if n_adm else 0.0,
            **overlap,
        }

        if len(Cg):
            q=Cg[
                Cg.admissible.astype(bool)
                & Cg.geometry_state_resolved.astype(bool)
                & np.isfinite(Cg.geometry_pair_cost)
            ]
            if len(q):
                row.update({
                    "geometry_pair_cost_median":float(q.geometry_pair_cost.median()),
                    "geometry_pair_cost_q90":float(q.geometry_pair_cost.quantile(.90)),
                    "alpha_cost_median":float(q.alpha_cost.median()),
                    "directional_correction_median":float(q.directional_correction.median()),
                    "directional_correction_abs_q90":float(q.directional_correction.abs().quantile(.90)),
                    "geometry_reversal_ratio_q90":float(q.geometry_reversal_ratio.quantile(.90)),
                    "geometry_reversal_ratio_q99":float(q.geometry_reversal_ratio.quantile(.99)),
                })

        if stop_reason:
            row["n_nodes_after"]=int(len(node_ids))
            row["n_selected"]=0
            row["stop_reason"]=stop_reason
            level_rows.append(row)
            break

        selected=GM[GM.selected_geometry].copy()
        selected["level"]=level
        selected["survivor_node"]=np.minimum(
            selected.super_i.astype(np.int64),
            selected.super_j.astype(np.int64)
        )
        selected["removed_node"]=np.maximum(
            selected.super_i.astype(np.int64),
            selected.super_j.astype(np.int64)
        )
        merge_parts.append(selected)

        selected_keys=set(
            zip(selected.super_i.astype(int),selected.super_j.astype(int))
        )
        Csave=Cg.copy()
        Csave["level"]=level
        Csave["selected_geometry"]=[
            (int(a),int(b)) in selected_keys
            for a,b in zip(Csave.super_i,Csave.super_j)
        ]
        # Unit-kappa correction allows later scale-sensitivity re-ranking without
        # reconstructing biology.
        step3global=json.loads(require(
            project/"results"/"hierarchy_v075_local_geometry"/"step3_global_bound"/
            "global_bound_global_certificate.json"
        ).read_text())
        kappa=float(step3global["global_scale_kappa"])
        Csave["directional_correction_per_unit_kappa"] = (
            Csave.directional_correction / kappa
        )
        boundary_parts.append(Csave)

        labels=contract_labels(labels,GM)
        row["n_nodes_after"]=int(len(np.unique(labels)))
        row["n_selected"]=n_sel
        level_rows.append(row)

    else:
        stop_reason="max_level_safety_ceiling"

    levels=pd.DataFrame(level_rows)
    bounds=pd.concat(boundary_parts,ignore_index=True) if boundary_parts else pd.DataFrame()
    merges=pd.concat(merge_parts,ignore_index=True) if merge_parts else pd.DataFrame()

    writepq(levels,out/"level_statistics.parquet")
    writepq(bounds,out/"boundary_geometry_all_levels.parquet")
    writepq(merges,out/"merge_trajectory.parquet")
    writepq(
        pd.DataFrame({
            "cell_index":np.arange(len(labels),dtype=np.int64),
            "hierarchy_node":labels.astype(np.int64),
        }),
        out/"final_membership.parquet"
    )

    # Figure-ready compact archive.
    if len(bounds):
        cols=[
            c for c in [
                "level","super_i","super_j","n_boundary_edges",
                "admissible","geometry_state_resolved",
                "alpha_cost","directional_correction",
                "directional_correction_per_unit_kappa",
                "geometry_pair_cost",
                "local_forward_cost","local_reverse_cost",
                "geometry_reversal_ratio",
                "molecular_distance","mechanics_support_fraction",
                "abs_tension_z","abs_delta_p_z",
                "comm_support","comm_directionality","comm_asymmetry",
                "selected_geometry",
            ] if c in bounds.columns
        ]
        writepq(bounds[cols],out/"figure_ready_geometry_trajectory.parquet")

    natural=stop_reason in {
        "no_remaining_supernode_boundaries",
        "no_biologically_admissible_boundaries",
        "no_disjoint_geometry_resolved_contractions",
    }

    # Geometry-resolution arrest is diagnostic HOLD rather than natural PASS.
    if stop_reason=="no_geometry_resolved_admissible_boundaries":
        natural=False

    report={
        "sample":sample,
        "n_level0_cells":int(len(cells)),
        "levels_executed":int(levels.level.max()) if len(levels) else 0,
        "total_geometry_ordered_contractions":int(len(merges)),
        "final_nodes":int(len(np.unique(labels))),
        "fraction_nodes_removed":float(
            (len(cells)-len(np.unique(labels)))/len(cells)
        ),
        "stop_reason":stop_reason,
        "natural_exhaustion":bool(natural),
        "pressure_reconstruction":pressure_audit,
        "geometry_policy":{
            "biology_gate_unchanged":True,
            "geometry_used_only_for_ordering":True,
            "pair_cost":"0.5*(F_A(z_B-z_A)+F_B(z_A-z_B))",
            "pair_cost_orientation_invariant":True,
            "directionality_retained":True,
            "supernode_covector":"cell-count weighted mean",
            "supernode_state":"mean molecular + mean observed mechanics state",
        },
        "status":"PASS" if natural else "HOLD",
    }
    (out/"geometry_aware_hierarchy_summary.json").write_text(
        json.dumps(report,indent=2)+"\n"
    )
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--max-levels",type=int,default=250)
    a=ap.parse_args()

    project=Path(a.project_root).resolve()
    source=require(
        project/"results"/"hierarchy_v075_local_geometry"/"step4_directional_norm"/
        "directional_norm_global_certificate.json"
    )
    d=json.loads(source.read_text())
    if d.get("READY_FOR_GEOMETRY_AWARE_HIERARCHY") is not True:
        raise SystemExit("ERROR: v0.7.5 local directional geometry not certified")

    print("STRATA 0.7.6 | Geometry-aware hierarchy")
    print("Biological admissibility is unchanged.")
    print("Certified direction-dependent local geometry orders admissible contractions.")
    print("Node mechanics: incident tension + pressure potential from certified delta-p constraints.")
    print("Orientation-invariant round-trip pair cost.")
    print(f"Parallel specimens: {min(a.workers,len(SAMPLES))}\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(SAMPLES))) as ex:
        futs={
            ex.submit(one,str(project),s,a.max_levels):s
            for s in SAMPLES
        }
        for f in as_completed(futs):
            r=f.result();reports.append(r)
            print(
                f"[DONE] {r['sample']}: "
                f"levels={r['levels_executed']} "
                f"merges={r['total_geometry_ordered_contractions']:,} "
                f"nodes={r['n_level0_cells']:,}->{r['final_nodes']:,} "
                f"removed={100*r['fraction_nodes_removed']:.2f}% "
                f"stop={r['stop_reason']} {r['status']}",
                flush=True
            )

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)

    out=project/"results"/"hierarchy_v076_geometry_aware"
    cert={
        "strata_version":"0.7.6",
        "stage":"geometry-aware hierarchy",
        "source_v075_step4_certificate_sha256":sha256_file(source),
        "policy":{
            "biological_admissibility_unchanged":True,
            "geometry_replaces_biology_gate":False,
            "geometry_orders_only_admissible_merges":True,
            "orientation_invariant_pair_cost":True,
            "actual_mechanical_state_coordinates_used":True,
            "pressure_state_from_certified_delta_p_constraints":True,
            "statistics_archive_complete":True,
        },
        "sample_reports":reports,
        "GEOMETRY_AWARE_HIERARCHY_GATE":"PASS" if gate else "HOLD",
        "GEOMETRY_DRIVES_HIERARCHY":bool(gate),
        "READY_FOR_DISCRETE_TRANSPORT":bool(gate),
    }
    p=out/"geometry_aware_hierarchy_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print(f"\nGEOMETRY-AWARE HIERARCHY GATE: {cert['GEOMETRY_AWARE_HIERARCHY_GATE']}")
    print(f"GEOMETRY DRIVES HIERARCHY: {cert['GEOMETRY_DRIVES_HIERARCHY']}")
    print(f"READY FOR DISCRETE TRANSPORT: {cert['READY_FOR_DISCRETE_TRANSPORT']}")
    print(f"Certificate: {p}")


if __name__=="__main__":
    main()
