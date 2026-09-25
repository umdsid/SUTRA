from __future__ import annotations
import argparse,json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from strata_hierarchy.v071.block_preflight import read_10x_cells_by_genes,normalized_expression_dense
from strata_hierarchy.v071.resource_intake import sha256_file
from strata_hierarchy.v076.geometry_hierarchy import aggregate_covectors
from strata_hierarchy.v092.sweep import (
    aggregate_completed_edges,completed_pair_cost,greedy_disjoint,apply_batch
)
from strata_hierarchy.v093.flow import (
    weighted_supernode_xy,aggregate_expression,topology_stats,
    summarize_array,shannon_sizes,heavy_geometry_snapshot
)
from strata_hierarchy.v093.ledger import ProductionLedger
from strata_hierarchy.v100.adaptive import (
    calibration_from_v094,online_metric,classify_online,
    choose_fraction,RegimeDetector
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def req(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p

def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def current_node_state(node_ids,xy_node,sizes,Ynode):
    return pd.DataFrame({
        "supernode_id":node_ids.astype(np.int64),
        "size":sizes.astype(np.int64),
        "centroid_x":xy_node[:,0],
        "centroid_y":xy_node[:,1],
        "expression_mean":Ynode.mean(axis=1),
        "expression_variance":Ynode.var(axis=1),
    })

def micro_summary(step,node_ids,sizes,se,selected,removed,fraction,mode,classification):
    row={
        "microstep":int(step),
        "nodes":int(len(node_ids)),
        "removed_fraction":float(removed),
        "batch_fraction":float(fraction),
        "batch_mode":str(mode),
        "selected_merges":int(len(selected)),
        "superedges":int(len(se)),
        "supernode_size_entropy":shannon_sizes(sizes),
        "supernode_size_mean":float(np.mean(sizes)),
        "supernode_size_max":int(np.max(sizes)),
        "online_state":classification["state"],
        "online_stable_support":float(classification["stable_support"]),
        "online_high_support":float(classification["high_support"]),
        "online_velocity":float(classification["median_velocity"]) if np.isfinite(classification["median_velocity"]) else np.nan,
        "online_acceleration":float(classification["median_acceleration"]) if np.isfinite(classification["median_acceleration"]) else np.nan,
    }
    row.update(topology_stats(se,node_ids))
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
    return row

def make_heavy(sample,landmark_idx,step,n0,labels,edges,xy,Y,Bcell,G,cfg):
    node_ids=np.unique(labels)
    se=aggregate_completed_edges(edges,labels)
    if len(se):
        se=completed_pair_cost(se,cfg["weights"])
    xy2,sizes2=weighted_supernode_xy(xy,labels,node_ids)
    Y2=aggregate_expression(Y,labels,node_ids)
    Bnode=aggregate_covectors(Bcell,labels,node_ids)

    removed=(n0-len(node_ids))/n0
    heavy={
        "sample":sample,
        "landmark_index":int(landmark_idx),
        "microstep":int(step),
        "removed_fraction":float(removed),
        "nodes":int(len(node_ids)),
        "superedges":int(len(se)),
        "ell":float(np.log(n0/max(len(node_ids),1))),
        "expression_total_variance":float(np.var(Y2,axis=0).sum()),
        "supernode_size_entropy":shannon_sizes(sizes2),
    }
    transport=pd.DataFrame()
    if len(se):
        heavy.update(topology_stats(se,node_ids))
        for c,p in [
            ("functional_distance","functional_distance"),
            ("cellchat_total","cellchat_total"),
            ("cellchat_directionality","cellchat_directionality"),
            ("tension_complete","tension_complete"),
            ("tension_confidence","tension_confidence"),
            ("delta_p_potential_complete","pressure_difference"),
            ("pressure_confidence","pressure_confidence"),
        ]:
            if c in se.columns: heavy.update(summarize_array(se[c],p))
        gstats,transport=heavy_geometry_snapshot(
            se,node_ids,xy2,Bnode,G,
            triangle_limit=int(cfg["triangle_limit_per_landmark"])
        )
        heavy.update(gstats)

    nstate=current_node_state(node_ids,xy2,sizes2,Y2)
    return heavy,se,nstate,transport

def one(project_s,sample,cfg):
    os.environ.update(
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1"
    )
    project=Path(project_s); t0=time.time()

    v91=project/"results"/"hierarchy_v091_network_exhaustive_completion"/sample
    edges=pd.read_parquet(req(v91/"completed_tissue_backbone_edges.parquet"))
    c91=json.loads(req(v91/"network_exhaustive_completion_certificate.json").read_text())
    n0=int(c91["n_cells"])

    # Frozen calibration from the exhaustive-run audit.
    v94=project/"results"/"hierarchy_v094_ness_flow_diagnostics"/sample
    v941=project/"results"/"hierarchy_v0941_flow_metric_audit"/sample
    norm=pd.read_parquet(req(v94/"observable_normalization.parquet"))
    audited=pd.read_parquet(req(v941/"block_balanced_flow.parquet"))
    norm,thresholds=calibration_from_v094(
        norm,audited,min_nodes=int(cfg["detector"]["min_effective_nodes"])
    )

    # Level-0 expression / coordinates.
    l0=project/"results"/"hierarchy_level0_v070"/sample
    manifest=json.loads(req(l0/"expression_manifest.json").read_text())
    matrix=Path(manifest["source_matrix"])
    if sha256_file(matrix)!=manifest["source_matrix_sha256"]:
        raise RuntimeError(f"{sample}: expression source hash changed")
    X,barcodes,genes=read_10x_cells_by_genes(matrix)
    Y=normalized_expression_dense(X).astype(np.float32)

    nodes0=pd.read_parquet(req(v91/"completed_tissue_node_fields.parquet"))
    xcol=next(c for c in ["centroid_x","x_centroid","center_x","x_center","x_location","x_coord","x"] if c in nodes0.columns)
    ycol=next(c for c in ["centroid_y","y_centroid","center_y","y_center","y_location","y_coord","y"] if c in nodes0.columns)
    xy=nodes0[[xcol,ycol]].to_numpy(float)

    # Frozen directional local geometry.
    s1=project/"results"/"hierarchy_v075_local_geometry"/"step1_symmetric_base"/sample
    s3=project/"results"/"hierarchy_v075_local_geometry"/"step3_global_bound"/sample
    G=sparse.load_npz(req(s1/"symmetric_base_metric.npz")).toarray()
    Bcell=sparse.load_npz(req(s3/"directional_covectors_scaled.npz")).tocsr()

    outroot=project/"results"/"hierarchy_v100_adaptive_ness_flow"
    ledger=ProductionLedger(outroot/"ledger",sample)
    out=outroot/sample; out.mkdir(parents=True,exist_ok=True)

    labels=np.arange(n0,dtype=np.int64)
    detector=RegimeDetector(cfg["detector"])
    heavy_rows=[]
    detector_rows=[]
    micro_rows=[]
    landmark_idx=0
    next_landmark=float(cfg["landmark_reduction_fraction"])
    total_merges=0
    stop_reason=None
    chosen_landmark=None

    # Initial heavy state (Level-0) for causal derivatives.
    heavy,se,nstate,tr=make_heavy(sample,landmark_idx,0,n0,labels,edges,xy,Y,Bcell,G,cfg)
    heavy_rows.append(heavy)
    ledger.landmark(landmark_idx,heavy,se,nstate,tr)
    landmark_idx+=1
    ledger.checkpoint(0,labels)

    classification={"stable_support":0.,"high_support":0.,"median_velocity":np.nan,
                    "median_acceleration":np.nan,"state":"WARMUP"}

    for step in range(int(cfg["max_microsteps_safety"])):
        node_ids=np.unique(labels)
        removed=(n0-len(node_ids))/n0
        if len(node_ids)<=int(cfg["detector"]["min_effective_nodes"]):
            stop_reason="minimum_tessellation_guard_reached_without_confirmed_regime"
            break

        se=aggregate_completed_edges(edges,labels)
        if len(se)==0:
            stop_reason="backbone_exhausted_without_confirmed_regime"
            break
        se=completed_pair_cost(se,cfg["weights"])

        frac,mode=choose_fraction(
            classification,cfg["step_control"],confirming=detector.confirming
        )
        cap=max(1,int(np.ceil(len(node_ids)*frac)))
        cap=min(cap,len(node_ids)//2)
        selected=greedy_disjoint(se,cap)
        if len(selected)==0:
            stop_reason="no_disjoint_contraction_batch"
            break

        xy_node,sizes=weighted_supernode_xy(xy,labels,node_ids)
        row=micro_summary(
            step,node_ids,sizes,se,selected,removed,frac,mode,classification
        )
        ledger.merges(step,selected)
        labels=apply_batch(labels,selected)
        total_merges+=len(selected)
        removed_after=(n0-len(np.unique(labels)))/n0
        row["nodes_after"]=int(len(np.unique(labels)))
        row["removed_fraction_after"]=float(removed_after)
        ledger.append_step(row); micro_rows.append(row)

        # Heavy states are reduction-spaced, not time-spaced.
        if removed_after+1e-15>=next_landmark:
            heavy,se2,nstate,tr=make_heavy(
                sample,landmark_idx,step+1,n0,labels,edges,xy,Y,Bcell,G,cfg
            )
            heavy_rows.append(heavy)
            ledger.landmark(landmark_idx,heavy,se2,nstate,tr)

            metric=online_metric(
                heavy_rows,norm,
                strides=tuple(cfg["detector"]["derivative_strides"]),
                windows=tuple(cfg["detector"]["causal_smoothing_windows"])
            )
            classification=classify_online(metric,thresholds)
            action=detector.update(
                landmark_idx,classification,
                nodes=int(heavy["nodes"]),
                removed=float(heavy["removed_fraction"])
            )
            dr={
                "landmark_index":landmark_idx,
                "microstep":step+1,
                "nodes":int(heavy["nodes"]),
                "removed_fraction":float(heavy["removed_fraction"]),
                "ell":float(heavy["ell"]),
                **classification,
                "detector_action":action,
                "confirming":bool(detector.confirming),
                "low_run":int(detector.low_run),
                "confirm_run":int(detector.confirm_run),
            }
            detector_rows.append(dr)

            if action=="STOP_CONFIRMED_REGIME":
                stop_reason="confirmed_persistent_low_flow_regime"
                chosen_landmark=int(detector.best_index if detector.best_index is not None else landmark_idx)
                ledger.checkpoint(step+1,labels)
                landmark_idx+=1
                break

            landmark_idx+=1
            next_landmark+=float(cfg["landmark_reduction_fraction"])

        if (step+1)%int(cfg["checkpoint_every_microsteps"])==0:
            ledger.checkpoint(step+1,labels)
    else:
        stop_reason="microstep_safety_ceiling"

    ledger.checkpoint(step+1,labels)

    mdf=pd.DataFrame(micro_rows)
    hdf=pd.DataFrame(heavy_rows)
    ddf=pd.DataFrame(detector_rows)
    wpq(mdf,out/"microstep_observables.parquet")
    wpq(hdf,out/"heavy_landmark_observables.parquet")
    wpq(ddf,out/"online_regime_detector.parquet")

    final_nodes=len(np.unique(labels))
    pass_stop=(stop_reason=="confirmed_persistent_low_flow_regime")
    summary={
        "sample":sample,
        "level0_cells":n0,
        "final_nodes":int(final_nodes),
        "total_merges":int(total_merges),
        "removed_fraction":float((n0-final_nodes)/n0),
        "microsteps":int(step+1),
        "heavy_landmarks":int(len(heavy_rows)),
        "stop_reason":stop_reason,
        "chosen_regime_landmark":chosen_landmark,
        "calibration_thresholds":thresholds,
        "detector":{
            "stable_support_required":cfg["detector"]["stable_support_required"],
            "persistence_landmarks":cfg["detector"]["persistence_landmarks"],
            "confirmation_landmarks":cfg["detector"]["confirmation_landmarks"],
            "minimum_effective_nodes":cfg["detector"]["min_effective_nodes"],
        },
        "elapsed_seconds":float(time.time()-t0),
        "status":"PASS" if pass_stop else "HOLD",
    }
    (out/"adaptive_ness_flow_certificate.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v100_adaptive_ness_flow.json")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())

    c=json.loads(req(
        project/"results"/"hierarchy_v0941_flow_metric_audit"/
        "flow_metric_audit_global_certificate.json"
    ).read_text())
    if c.get("READY_FOR_ADAPTIVE_NESS_FLOW_DESIGN") is not True:
        raise SystemExit("ERROR: v0.9.4.1 robust-regime audit not certified")

    print("STRATA 1.0.0 | Adaptive NESS multiscale flow")
    print("Three specimens run independently in parallel.")
    print("Base contraction fraction: 5e-4.")
    print("Sharp changes and low-flow confirmation use 2e-4.")
    print("No target node count and no exhaustive-collapse stopping rule.")
    print("Stop requires a persistent low-flow regime plus causal ultraslow confirmation.")
    print("Minimum node count is a tessellation guard only, never a target.")
    print("Full multiscale ledger remains enabled.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result();reports.append(r)
            print(
                f"[DONE] {r['sample']}: microsteps={r['microsteps']:,} "
                f"nodes={r['level0_cells']:,}->{r['final_nodes']:,} "
                f"removed={100*r['removed_fraction']:.2f}% "
                f"landmarks={r['heavy_landmarks']} "
                f"stop={r['stop_reason']} {r['status']}",
                flush=True
            )

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v100_adaptive_ness_flow"
    g={
        "strata_version":"1.0.0",
        "stage":"adaptive NESS multiscale flow",
        "parallel_specimens":3,
        "scientific_contract":{
            "target_node_count_used":False,
            "exhaustive_component_collapse_used_as_stop":False,
            "block_balanced_effective_state_flow":True,
            "causal_multiresolution_detector":True,
            "ultraslow_confirmation_required":True,
            "minimum_tessellation_guard":int(cfg["detector"]["min_effective_nodes"]),
            "full_tracking_contract_retained":True,
        },
        "sample_reports":reports,
        "ADAPTIVE_NESS_FLOW_GATE":"PASS" if gate else "HOLD",
        "SELF_STOPPING_EFFECTIVE_REGIMES_CERTIFIED":bool(gate),
        "READY_FOR_SPATIAL_DOMAIN_ATLAS":bool(gate),
    }
    p=out/"adaptive_ness_flow_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nADAPTIVE NESS FLOW GATE: {g['ADAPTIVE_NESS_FLOW_GATE']}")
    print(f"SELF-STOPPING EFFECTIVE REGIMES CERTIFIED: {g['SELF_STOPPING_EFFECTIVE_REGIMES_CERTIFIED']}")
    print(f"READY FOR SPATIAL DOMAIN ATLAS: {g['READY_FOR_SPATIAL_DOMAIN_ATLAS']}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
