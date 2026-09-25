from __future__ import annotations
import argparse,json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from sutra.hierarchy.v071.block_preflight import read_10x_cells_by_genes,normalized_expression_dense
from sutra.hierarchy.v071.resource_intake import sha256_file
from sutra.hierarchy.v076.geometry_hierarchy import aggregate_covectors
from sutra.hierarchy.v093.flow import (
    schedule_fraction,weighted_supernode_xy,aggregate_expression,
    topology_stats,summarize_array,shannon_sizes,derivative_table,
    heavy_geometry_snapshot
)
from sutra.hierarchy.v092.sweep import (
    aggregate_completed_edges,completed_pair_cost,greedy_disjoint,apply_batch
)
from sutra.hierarchy.v093.ledger import ProductionLedger

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def require(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p

def writepq(df,path):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),path,compression="zstd")

def current_node_state(labels,node_ids,xy_node,sizes,Ynode):
    return pd.DataFrame({
        "supernode_id":node_ids.astype(np.int64),
        "size":sizes.astype(np.int64),
        "centroid_x":xy_node[:,0],
        "centroid_y":xy_node[:,1],
        "expression_mean":Ynode.mean(axis=1),
        "expression_variance":Ynode.var(axis=1),
    })

def microstep_summary(step,n0,node_ids,sizes,se,selected,removed,fraction,mode):
    row={
        "microstep":int(step),
        "nodes":int(len(node_ids)),
        "removed_fraction":float(removed),
        "batch_fraction":float(fraction),
        "batch_mode":mode,
        "selected_merges":int(len(selected)),
        "superedges":int(len(se)),
        "component_count":0,
        "cycle_rank":0,
        "supernode_size_entropy":shannon_sizes(sizes),
        "supernode_size_mean":float(np.mean(sizes)),
        "supernode_size_max":int(np.max(sizes)),
    }
    topo=topology_stats(se,node_ids)
    row["component_count"]=topo["components"]
    row["cycle_rank"]=topo["cycle_rank"]
    row["mean_degree"]=topo["mean_degree"]
    row["max_degree"]=topo["max_degree"]

    for c,p in [
        ("completed_pair_cost","pair_cost"),
        ("expression_distance","expression_distance"),
        ("functional_distance","functional_distance"),
        ("cellchat_total","cellchat_total"),
        ("cellchat_directionality","cellchat_directionality"),
        ("cellchat_support_forward","cellchat_support_forward"),
        ("cellchat_support_reverse","cellchat_support_reverse"),
        ("tension_complete","tension_complete"),
        ("tension_confidence","tension_confidence"),
        ("delta_p_potential_complete","pressure_difference"),
        ("pressure_confidence","pressure_confidence"),
        ("spatial_distance","spatial_distance"),
        ("gap_guard_ratio","gap_guard_ratio"),
    ]:
        if c in se.columns:
            row.update(summarize_array(se[c],p))
    if "tension_provenance" in se.columns:
        vc=se.tension_provenance.value_counts(normalize=True)
        row["tension_anchor_supported_fraction"]=float(vc.get("anchor_supported",0.0))
        row["tension_harmonic_extension_fraction"]=float(vc.get("harmonic_extension",0.0))
        row["tension_neutral_prior_fraction"]=float(vc.get("neutral_prior",0.0))
    return row

def one(project_s,sample,cfg):
    os.environ.update(
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1"
    )
    t0=time.time(); project=Path(project_s)

    v91=project/"results"/"hierarchy_v091_network_exhaustive_completion"/sample
    edges=pd.read_parquet(require(v91/"completed_tissue_backbone_edges.parquet"))
    cert=json.loads(require(v91/"network_exhaustive_completion_certificate.json").read_text())
    n0=int(cert["n_cells"])

    # Frozen Level-0 expression and spatial state.
    l0=project/"results"/"hierarchy_level0_v070"/sample
    cells=pd.read_parquet(require(l0/"cells.parquet"))
    manifest=json.loads(require(l0/"expression_manifest.json").read_text())
    matrix=Path(manifest["source_matrix"])
    if sha256_file(matrix)!=manifest["source_matrix_sha256"]:
        raise RuntimeError(f"{sample}: expression source hash changed")
    X,barcodes,genes=read_10x_cells_by_genes(matrix)
    Y=normalized_expression_dense(X).astype(np.float32)

    # Use standard centroid columns from completed node fields, avoiding schema drift.
    nodes0=pd.read_parquet(require(v91/"completed_tissue_node_fields.parquet"))
    xcol=next(c for c in ["centroid_x","x_centroid","center_x","x_center","x_location","x_coord","x"] if c in nodes0.columns)
    ycol=next(c for c in ["centroid_y","y_centroid","center_y","y_center","y_location","y_coord","y"] if c in nodes0.columns)
    xy=nodes0[[xcol,ycol]].to_numpy(float)

    # Frozen directional geometry ingredients from Level-0.
    s1=project/"results"/"hierarchy_v075_local_geometry"/"step1_symmetric_base"/sample
    s3=project/"results"/"hierarchy_v075_local_geometry"/"step3_global_bound"/sample
    G=sparse.load_npz(require(s1/"symmetric_base_metric.npz")).toarray()
    Bcell=sparse.load_npz(require(s3/"directional_covectors_scaled.npz")).tocsr()

    labels=np.arange(n0,dtype=np.int64)
    ledger=ProductionLedger(
        project/"results"/"hierarchy_v093_full_completed_ultraslow_flow"/"ledger",
        sample
    )
    ledger.checkpoint(0,labels)

    weights=cfg["weights"]
    landmark_step=float(cfg["heavy_landmark_reduction_fraction"])
    next_landmark=0.0
    landmark_idx=0
    total_merges=0
    stop_reason=None
    micro_rows=[]

    max_steps=int(cfg["max_microsteps_safety"])
    for step in range(max_steps):
        node_ids=np.unique(labels)
        removed=(n0-len(node_ids))/n0

        se=aggregate_completed_edges(edges,labels)
        if len(se)==0 or len(node_ids)<=1:
            stop_reason="no_backbone_superedges" if len(se)==0 else "single_supernode"
            break

        se=completed_pair_cost(se,weights)
        frac,mode=schedule_fraction(removed,cfg)
        cap=max(1,int(np.ceil(len(node_ids)*frac)))
        cap=min(cap,len(node_ids)//2)
        selected=greedy_disjoint(se,cap)
        if len(selected)==0:
            stop_reason="no_disjoint_contraction_batch"
            break

        xy_node,sizes=weighted_supernode_xy(xy,labels,node_ids)
        Ynode=aggregate_expression(Y,labels,node_ids)

        row=microstep_summary(
            step,n0,node_ids,sizes,se,selected,removed,frac,mode
        )
        # expression/global state at every microstep
        row.update(summarize_array(Ynode.mean(axis=0),"gene_mean_across_supernodes"))
        row.update(summarize_array(Ynode.var(axis=0),"gene_variance_across_supernodes"))
        row["expression_total_variance"]=float(np.var(Ynode,axis=0).sum())
        row["functional_similarity_mean"]=float(se.functional_similarity.mean()) if "functional_similarity" in se.columns else np.nan
        row["cellchat_forward_mean"]=float(se.cellchat_forward.mean()) if "cellchat_forward" in se.columns else np.nan
        row["cellchat_reverse_mean"]=float(se.cellchat_reverse.mean()) if "cellchat_reverse" in se.columns else np.nan

        ledger.merges(step,selected)
        labels=apply_batch(labels,selected)
        total_merges += len(selected)
        removed_after=(n0-len(np.unique(labels)))/n0
        row["removed_fraction_after"]=float(removed_after)
        row["nodes_after"]=int(len(np.unique(labels)))
        ledger.append_step(row)
        micro_rows.append(row)

        # Dense heavy landmarks across scale.
        while removed_after+1e-15 >= next_landmark:
            node_ids2=np.unique(labels)
            se2=aggregate_completed_edges(edges,labels)
            if len(se2):
                se2=completed_pair_cost(se2,weights)
            xy2,sizes2=weighted_supernode_xy(xy,labels,node_ids2)
            Y2=aggregate_expression(Y,labels,node_ids2)
            Bnode=aggregate_covectors(Bcell,labels,node_ids2)

            heavy={
                "sample":sample,
                "landmark_index":landmark_idx,
                "microstep":int(step),
                "removed_fraction":float(removed_after),
                "nodes":int(len(node_ids2)),
                "superedges":int(len(se2)),
                "expression_total_variance":float(np.var(Y2,axis=0).sum()),
                "supernode_size_entropy":shannon_sizes(sizes2),
            }
            if len(se2):
                heavy.update(topology_stats(se2,node_ids2))
                heavy.update(summarize_array(se2.functional_distance,"functional_distance"))
                heavy.update(summarize_array(se2.cellchat_total,"cellchat_total"))
                heavy.update(summarize_array(se2.cellchat_directionality,"cellchat_directionality"))
                heavy.update(summarize_array(se2.tension_complete,"tension_complete"))
                heavy.update(summarize_array(se2.tension_confidence,"tension_confidence"))
                heavy.update(summarize_array(se2.delta_p_potential_complete,"pressure_difference"))
                heavy.update(summarize_array(se2.pressure_confidence,"pressure_confidence"))
                gstats,transport=heavy_geometry_snapshot(
                    se2,node_ids2,xy2,Bnode,G,
                    triangle_limit=int(cfg["triangle_limit_per_landmark"])
                )
                heavy.update(gstats)
            else:
                transport=pd.DataFrame()

            nstate=current_node_state(labels,node_ids2,xy2,sizes2,Y2)
            ledger.landmark(
                landmark_idx,heavy,
                superedges=se2,node_state=nstate,transport=transport
            )
            landmark_idx+=1
            next_landmark += landmark_step

        if step%int(cfg["checkpoint_every_microsteps"])==0:
            ledger.checkpoint(step+1,labels)
    else:
        stop_reason="microstep_safety_ceiling"

    ledger.checkpoint(step+1,labels)

    mdf=pd.DataFrame(micro_rows)
    if len(mdf):
        deriv=derivative_table(mdf,"removed_fraction_after")
        outdir=project/"results"/"hierarchy_v093_full_completed_ultraslow_flow"/sample
        outdir.mkdir(parents=True,exist_ok=True)
        writepq(mdf,outdir/"microstep_observables.parquet")
        writepq(deriv,outdir/"microstep_observables_with_derivatives.parquet")
    else:
        outdir=project/"results"/"hierarchy_v093_full_completed_ultraslow_flow"/sample
        outdir.mkdir(parents=True,exist_ok=True)

    final_nodes=len(np.unique(labels))
    natural=stop_reason in ("no_backbone_superedges","single_supernode")
    summary={
        "sample":sample,
        "level0_cells":n0,
        "final_nodes":int(final_nodes),
        "total_merges":int(total_merges),
        "removed_fraction":float((n0-final_nodes)/n0),
        "microsteps":int(step+1),
        "heavy_landmarks":int(landmark_idx),
        "stop_reason":stop_reason,
        "natural_exhaustion":bool(natural),
        "elapsed_seconds":float(time.time()-t0),
        "status":"PASS" if natural else "HOLD",
    }
    (outdir/"full_completed_flow_certificate.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v093_full_completed_ultraslow_flow.json")
    a=ap.parse_args()

    project=Path(a.project_root).resolve()
    cfg=json.loads(require(project/a.config).read_text())

    s92=require(project/"results"/"hierarchy_v092_step_spectrum_consistency"/"step_spectrum_global_certificate.json")
    d92=json.loads(s92.read_text())
    if d92.get("READY_TO_SELECT_PRODUCTION_STEP") is not True:
        raise SystemExit("ERROR: v0.9.2 slow-limit consistency is not certified")

    s91=require(project/"results"/"hierarchy_v091_network_exhaustive_completion"/"network_exhaustive_completion_global_certificate.json")
    d91=json.loads(s91.read_text())
    if d91.get("READY_FOR_COMPLETED_ULTRASLOW_FLOW") is not True:
        raise SystemExit("ERROR: v0.9.1 completed tissue state is not ready")

    print("STRATA 0.9.3 | Full completed ultraslow multiscale flow")
    print("Three specimens run in parallel.")
    print("Production starts at validated fraction 2e-4.")
    print("Larger batches appear only after substantial reduction; scientific model is fixed.")
    print("No 10% pilot ceiling: run continues to natural backbone exhaustion.")
    print("Full observable ledger + dense transport/holonomy landmarks + derivatives enabled.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        futs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(futs):
            r=f.result(); reports.append(r)
            print(
                f"[DONE] {r['sample']}: microsteps={r['microsteps']:,} "
                f"merges={r['total_merges']:,} "
                f"nodes={r['level0_cells']:,}->{r['final_nodes']:,} "
                f"removed={100*r['removed_fraction']:.2f}% "
                f"landmarks={r['heavy_landmarks']:,} "
                f"{r['stop_reason']} {r['status']}",
                flush=True
            )

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v093_full_completed_ultraslow_flow"
    cert={
        "strata_version":"0.9.3",
        "stage":"full completed ultraslow multiscale flow",
        "parallel_specimens":3,
        "tracking_contract":{
            "expression":True,
            "GO_MSIGDB_functional_state":True,
            "CellChat":True,
            "mechanics_confidence_and_provenance":True,
            "pressure":True,
            "topology":True,
            "local_geometry":True,
            "directional_state":True,
            "transport":True,
            "geodesics":True,
            "holonomy":True,
            "holonomy_density":True,
            "component_structure":True,
            "merger_ancestry":True,
            "first_scale_differences":True,
            "second_scale_differences":True,
        },
        "sample_reports":reports,
        "FULL_COMPLETED_FLOW_GATE":"PASS" if gate else "HOLD",
        "FULL_MULTISCALE_TRAJECTORY_COMPLETE":bool(gate),
        "READY_FOR_FLOW_ATLAS_AND_MANUSCRIPT_STATISTICS":bool(gate),
    }
    p=out/"full_completed_flow_global_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print(f"\nFULL COMPLETED FLOW GATE: {cert['FULL_COMPLETED_FLOW_GATE']}")
    print(f"FULL MULTISCALE TRAJECTORY COMPLETE: {cert['FULL_MULTISCALE_TRAJECTORY_COMPLETE']}")
    print(f"READY FOR FLOW ATLAS AND MANUSCRIPT STATISTICS: {cert['READY_FOR_FLOW_ATLAS_AND_MANUSCRIPT_STATISTICS']}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
