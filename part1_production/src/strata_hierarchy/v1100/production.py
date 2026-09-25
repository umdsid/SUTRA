from __future__ import annotations
import json, os, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

from strata_hierarchy.v071.block_preflight import (
    read_10x_cells_by_genes, normalized_expression_dense
)
from strata_hierarchy.v071.resource_intake import sha256_file
from strata_hierarchy.v076.geometry_hierarchy import aggregate_covectors
from strata_hierarchy.v092.sweep import (
    aggregate_completed_edges, completed_pair_cost, greedy_disjoint, apply_batch
)
from strata_hierarchy.v093.flow import (
    schedule_fraction, weighted_supernode_xy, aggregate_expression,
    topology_stats, summarize_array, shannon_sizes, heavy_geometry_snapshot
)
from strata_hierarchy.v110.terminal_rule import (
    evaluate_once, evaluate_persistent, canonical_json_hash, validate_frozen_config
)
from strata_hierarchy.v1100.invariants import (
    mass_counts, assert_disjoint_selected, assert_batch_exact,
    level0_components, active_component_map, assert_merge_components,
    LineageTracker, assert_lineage_matches_labels
)
from strata_hierarchy.v1100.metrics import (
    mass_statistics, node_coherence, attach_block_speeds,
    add_derivatives, json_safe
)
from strata_hierarchy.v1100.ledger import (
    FrozenProductionLedger, writepq, writejson
)

EXPECTED_TERMINAL_RULE_SHA256="06c99faaa3733ad6129e982804698cd3731388e02a6bf8e879679a3c8eb4accc"
EXPECTED_TERMINAL_RULE_CODE_SHA256="dd19488a5baca907fcf97c85bfef8e7349b9a68c15af2551602954e8a12cceaf"

def require(p):
    p=Path(p)
    if not p.exists():raise FileNotFoundError(p)
    return p

def file_sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def verify_freeze(project):
    freeze_dir=project/"results"/"hierarchy_v110_production_terminal_rule_freeze"
    cert=json.loads(require(
        freeze_dir/"production_terminal_rule_freeze_global_certificate.json"
    ).read_text())
    if cert.get("READY_FOR_FINAL_PRODUCTION_RERUN") is not True:
        raise RuntimeError("v1.0.10 does not certify final production rerun")
    rule_path=require(freeze_dir/"production_terminal_rule_FROZEN.json")
    rule=json.loads(rule_path.read_text())
    errs=validate_frozen_config(rule)
    if errs:raise RuntimeError("frozen terminal rule invalid: "+"; ".join(errs))
    rh=canonical_json_hash(rule)
    if rh!=EXPECTED_TERMINAL_RULE_SHA256 or rh!=cert.get("terminal_rule_contract_sha256"):
        raise RuntimeError(f"terminal-rule hash mismatch: {rh}")
    code_path=require(project/"src"/"strata_hierarchy"/"v110"/"terminal_rule.py")
    ch=file_sha(code_path)
    if ch!=EXPECTED_TERMINAL_RULE_CODE_SHA256:
        raise RuntimeError(
            "v1.0.10 terminal_rule.py changed after inspected freeze "
            f"({ch} != {EXPECTED_TERMINAL_RULE_CODE_SHA256})"
        )

    manifest=require(freeze_dir/"scientific_source_hash_manifest.parquet")
    m=pd.read_parquet(manifest)
    changed=[]
    for r in m.itertuples(index=False):
        p=project/str(r.relative_path)
        if not p.exists() or file_sha(p)!=str(r.sha256):
            changed.append(str(r.relative_path))
    if changed:
        raise RuntimeError(
            "scientific source/config hash drift since v1.0.10 freeze: "
            +", ".join(changed[:20])
        )
    return rule,cert

def categorical_provenance(edges,labels,superedges):
    if len(superedges)==0:return superedges
    cat_cols=[
        c for c in ["tension_provenance","pressure_provenance","mechanics_provenance"]
        if c in edges.columns
    ]
    if not cat_cols:return superedges
    lab=np.asarray(labels,np.int64)
    i0=edges.cell_i_index.to_numpy(np.int64)
    j0=edges.cell_j_index.to_numpy(np.int64)
    si=lab[i0];sj=lab[j0]
    q=si!=sj
    if not q.any():return superedges
    x=edges.loc[q,cat_cols].copy()
    x["super_i"]=np.minimum(si[q],sj[q])
    x["super_j"]=np.maximum(si[q],sj[q])
    for c in cat_cols:
        # Deterministic modal provenance; lexical tie break.
        z=(x.groupby(["super_i","super_j"])[c]
             .agg(lambda s: sorted(
                 s.astype(str).value_counts()[lambda a:a==a.max()].index
             )[0])
             .reset_index())
        superedges=superedges.merge(z,on=["super_i","super_j"],how="left")
    return superedges

def current_node_state(
    node_ids,xy_node,sizes,Ynode,expr_mse,spatial_rms,dominant_mask,lineage
):
    return pd.DataFrame({
        "supernode_id":node_ids.astype(np.int64),
        "mass":sizes.astype(np.int64),
        "mass_fraction":sizes/sizes.sum(),
        "centroid_x":xy_node[:,0],
        "centroid_y":xy_node[:,1],
        "expression_mean":Ynode.mean(axis=1),
        "expression_variance":Ynode.var(axis=1),
        "within_expression_mse":expr_mse,
        "spatial_rms_radius":spatial_rms,
        "dominant_mass_set":dominant_mask.astype(bool),
        "lineage_token":[lineage.token_by_supernode[int(u)] for u in node_ids],
    })

def evaluation_observables(
    sample,eval_idx,step,n0,labels,node_ids,se,xy_node,sizes,Ynode,
    Bnode,G,cell_expr_sqmean,cell_r2,triangle_limit
):
    row={
        "sample":sample,
        "evaluation_index":int(eval_idx),
        "microstep":int(step),
        "nodes":int(len(node_ids)),
        "removed_fraction":float((n0-len(node_ids))/n0),
        "superedges":int(len(se)),
        "supernode_size_entropy":shannon_sizes(sizes),
        "supernode_size_mean":float(np.mean(sizes)),
        "supernode_size_max":int(np.max(sizes)),
        "expression_total_variance":float(np.var(Ynode,axis=0).sum()),
    }
    row.update(mass_statistics(sizes))

    expr_mse,spatial_rms,dexpr,dspatial,dom=node_coherence(
        labels,node_ids,sizes,Ynode,xy_node,cell_expr_sqmean,cell_r2
    )
    row["dominant_expression_coherence"]=dexpr
    row["dominant_spatial_coherence"]=dspatial

    row.update(summarize_array(Ynode.mean(axis=0),"gene_mean_across_supernodes"))
    row.update(summarize_array(Ynode.var(axis=0),"gene_variance_across_supernodes"))

    transport=pd.DataFrame()
    if len(se):
        row.update(topology_stats(se,node_ids))
        for c,prefix in [
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
                row.update(summarize_array(se[c],prefix))
        if "functional_similarity" in se.columns:
            row["functional_similarity_mean"]=float(se.functional_similarity.mean())
        if "tension_provenance" in se.columns:
            vc=se.tension_provenance.astype(str).value_counts(normalize=True)
            row["tension_anchor_supported_fraction"]=float(vc.get("anchor_supported",0.0))
            row["tension_harmonic_extension_fraction"]=float(vc.get("harmonic_extension",0.0))
            row["tension_neutral_prior_fraction"]=float(vc.get("neutral_prior",0.0))

        gstats,transport=heavy_geometry_snapshot(
            se,node_ids,xy_node,Bnode,G,triangle_limit=int(triangle_limit)
        )
        row.update(gstats)
        tri=float(row.get("triangle_candidates",0))
        row["triangle_resolved_fraction"]=(
            float(row.get("triangle_resolved",0))/tri if tri>0 else np.nan
        )
    else:
        row.update({
            "components":int(len(node_ids)),
            "cycle_rank":0,
            "mean_degree":0.0,
            "max_degree":0,
            "triangle_resolved_fraction":np.nan,
        })

    return row,transport,expr_mse,spatial_rms,dom

def microstep_row(step,n0,before_ids,after_ids,se,selected,removed_before,removed_after,fraction,mode):
    row={
        "microstep":int(step),
        "nodes_before":int(len(before_ids)),
        "nodes_after":int(len(after_ids)),
        "removed_fraction_before":float(removed_before),
        "removed_fraction_after":float(removed_after),
        "batch_fraction":float(fraction),
        "batch_mode":str(mode),
        "selected_merges":int(len(selected)),
        "candidate_superedges":int(len(se)),
        "partition_exact":True,
        "mass_exact":True,
    }
    if len(selected) and "completed_pair_cost" in selected.columns:
        row["selected_pair_cost_mean"]=float(selected.completed_pair_cost.mean())
        row["selected_pair_cost_q95"]=float(selected.completed_pair_cost.quantile(.95))
    return row

def run_one(project_s,sample,engine_cfg,rule,resume=False):
    os.environ.update(
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1"
    )
    t0=time.time()
    project=Path(project_s)

    v91=project/"results"/"hierarchy_v091_network_exhaustive_completion"/sample
    edges=pd.read_parquet(require(v91/"completed_tissue_backbone_edges.parquet"))
    cert=json.loads(require(v91/"network_exhaustive_completion_certificate.json").read_text())
    n0=int(cert["n_cells"])

    l0=project/"results"/"hierarchy_level0_v070"/sample
    _cells=pd.read_parquet(require(l0/"cells.parquet"))
    manifest=json.loads(require(l0/"expression_manifest.json").read_text())
    matrix=Path(manifest["source_matrix"])
    if sha256_file(matrix)!=manifest["source_matrix_sha256"]:
        raise RuntimeError(f"{sample}: Level-0 expression source hash changed")
    X,barcodes,genes=read_10x_cells_by_genes(matrix)
    Y=normalized_expression_dense(X).astype(np.float32)
    if Y.shape[0]!=n0:
        raise RuntimeError(f"{sample}: expression rows {Y.shape[0]} != N0 {n0}")

    nodes0=pd.read_parquet(require(v91/"completed_tissue_node_fields.parquet"))
    xcol=next(c for c in [
        "centroid_x","x_centroid","center_x","x_center","x_location","x_coord","x"
    ] if c in nodes0.columns)
    ycol=next(c for c in [
        "centroid_y","y_centroid","center_y","y_center","y_location","y_coord","y"
    ] if c in nodes0.columns)
    xy=nodes0[[xcol,ycol]].to_numpy(float)
    if len(xy)!=n0:
        raise RuntimeError(f"{sample}: coordinate rows {len(xy)} != N0 {n0}")

    cell_expr_sqmean=np.mean(Y.astype(np.float64)**2,axis=1)
    cell_r2=np.sum(xy*xy,axis=1)

    s1=project/"results"/"hierarchy_v075_local_geometry"/"step1_symmetric_base"/sample
    s3=project/"results"/"hierarchy_v075_local_geometry"/"step3_global_bound"/sample
    G=sparse.load_npz(require(s1/"symmetric_base_metric.npz")).toarray()
    Bcell=sparse.load_npz(require(s3/"directional_covectors_scaled.npz")).tocsr()

    level0_comp=level0_components(edges,n0)

    outroot=project/"results"/"hierarchy_v1100_frozen_production_rerun"
    sample_out=outroot/sample
    ledger=FrozenProductionLedger(outroot/"ledger",sample)

    eval_path=sample_out/"scientific_evaluations.parquet"
    derivative_path=sample_out/"scientific_evaluations_with_derivatives.parquet"
    sample_out.mkdir(parents=True,exist_ok=True)

    if resume:
        cp=ledger.latest_checkpoint()
    else:
        cp=None
        # Refuse accidental contamination from an earlier production attempt.
        if (ledger.root/"microsteps.jsonl").exists() and (ledger.root/"microsteps.jsonl").stat().st_size:
            raise RuntimeError(
                f"{sample}: existing production ledger found; use --resume or remove "
                f"{ledger.root}"
            )

    if cp is None:
        labels=np.arange(n0,dtype=np.int64)
        lineage=LineageTracker.level0(n0)
        start_step=0
        total_merges=0
        eval_idx=0
        next_eval=float(engine_cfg["heavy_landmark_reduction_fraction"])
        eval_df=pd.DataFrame()
        ledger.checkpoint(0,labels,lineage.active_frame(),{
            "next_step":0,"total_merges":0,"evaluation_index":0,
            "next_evaluation_removed_fraction":next_eval,
        })
    else:
        cp_step,state,labels,lineage_frame=cp
        lineage=LineageTracker.from_frame(lineage_frame)
        start_step=int(state["next_step"])
        total_merges=int(state["total_merges"])
        eval_idx=int(state["evaluation_index"])
        next_eval=float(state["next_evaluation_removed_fraction"])
        eval_df=pd.read_parquet(eval_path) if eval_path.exists() else pd.DataFrame()
        assert_lineage_matches_labels(lineage,labels,n0)

    weights=engine_cfg["weights"]
    max_steps=int(engine_cfg["max_microsteps_safety"])
    checkpoint_every=int(engine_cfg["checkpoint_every_microsteps"])
    eval_spacing=float(engine_cfg["heavy_landmark_reduction_fraction"])
    triangle_limit=int(engine_cfg["triangle_limit_per_landmark"])

    stop_reason=None
    status="HOLD"
    final_terminal={}
    initial_component_map=active_component_map(labels,level0_comp)

    for step in range(start_step,max_steps):
        before_ids,_=mass_counts(labels,n0)
        if len(before_ids)<=int(rule["minimum_tessellation_guard_nodes"]):
            stop_reason="minimum_tessellation_guard_reached_without_scientific_stop"
            break

        removed_before=(n0-len(before_ids))/n0
        se=aggregate_completed_edges(edges,labels)
        if len(se)==0 or len(before_ids)<=1:
            stop_reason="no_backbone_superedges_before_scientific_stop"
            break
        se=completed_pair_cost(se,weights)
        frac,mode=schedule_fraction(removed_before,engine_cfg)
        cap=max(1,int(np.ceil(len(before_ids)*frac)))
        cap=min(cap,len(before_ids)//2)
        selected=greedy_disjoint(se,cap)
        if len(selected)==0:
            stop_reason="no_disjoint_contraction_batch_before_scientific_stop"
            break

        comp_map=active_component_map(labels,level0_comp)
        assert_disjoint_selected(selected,before_ids)
        assert_merge_components(selected,comp_map)

        before=labels.copy()
        ancestry=lineage.apply(selected,step)
        labels=apply_batch(labels,selected)
        inv=assert_batch_exact(before,labels,selected,n0)
        assert_lineage_matches_labels(lineage,labels,n0)

        after_ids,_=mass_counts(labels,n0)
        total_merges+=len(selected)
        removed_after=(n0-len(after_ids))/n0
        mrow=microstep_row(
            step,n0,before_ids,after_ids,se,selected,
            removed_before,removed_after,frac,mode
        )
        ledger.merge_batch(step,selected,ancestry)
        ledger.append_step(mrow)

        # Scientific evaluations occur only at genuine post-batch states.
        crossed=removed_after+1e-15>=next_eval
        if crossed:
            # Advance all crossed storage thresholds without duplicating the same state.
            while removed_after+1e-15>=next_eval:
                next_eval += eval_spacing

            node_ids=np.unique(labels)
            se2=aggregate_completed_edges(edges,labels)
            if len(se2):
                se2=completed_pair_cost(se2,weights)
                se2=categorical_provenance(edges,labels,se2)
            xy2,sizes2=weighted_supernode_xy(xy,labels,node_ids)
            Y2=aggregate_expression(Y,labels,node_ids)
            Bnode=aggregate_covectors(Bcell,labels,node_ids)

            raw,transport,expr_mse,spatial_rms,dom=evaluation_observables(
                sample,eval_idx,step,n0,labels,node_ids,se2,xy2,sizes2,Y2,
                Bnode,G,cell_expr_sqmean,cell_r2,triangle_limit
            )

            previous=eval_df.iloc[-1].to_dict() if len(eval_df) else None
            row=attach_block_speeds(previous,raw)
            eval_df=pd.concat([eval_df,pd.DataFrame([row])],ignore_index=True)
            deriv=add_derivatives(eval_df,"removed_fraction")
            writepq(eval_df,eval_path)
            writepq(deriv,derivative_path)

            once=evaluate_once(eval_df,rule)
            persistent=evaluate_persistent(eval_df,rule)
            final_terminal=persistent
            block_table=once.get("block_table",pd.DataFrame())

            nstate=current_node_state(
                node_ids,xy2,sizes2,Y2,expr_mse,spatial_rms,dom,lineage
            )
            term_json=json_safe({
                "once":once,
                "persistent":persistent,
                "frozen_terminal_rule_sha256":EXPECTED_TERMINAL_RULE_SHA256,
            })
            ledger.evaluation(
                eval_idx,step,labels,nstate,se2,transport,block_table,
                term_json,json_safe(row),lineage.active_frame(),
                derivative_path,rule["required_checkpoint_payload"]
            )

            last_once=persistent.get("evaluations",[])
            last_once=last_once[-1] if last_once else once
            print(
                f"[{sample}] eval={eval_idx:03d} step={step:,} "
                f"nodes={len(node_ids):,} removed={100*removed_after:6.2f}% "
                f"slow={last_once.get('slow_block_fraction',0.0):.2f} "
                f"mass={'PASS' if last_once.get('mass_stable',False) else 'HOLD'} "
                f"expr={'PASS' if last_once.get('expression_stable',False) else 'HOLD'} "
                f"spatial={'PASS' if last_once.get('spatial_stable',False) else 'HOLD'} "
                f"confirm={persistent.get('confirmations',0)}/"
                f"{rule['persistence_confirmations']}",
                flush=True
            )

            eval_idx+=1

            run_state={
                "next_step":int(step+1),
                "total_merges":int(total_merges),
                "evaluation_index":int(eval_idx),
                "next_evaluation_removed_fraction":float(next_eval),
            }
            ledger.checkpoint(step+1,labels,lineage.active_frame(),run_state)

            if persistent.get("stop") is True:
                stop_reason="confirmed_persistent_terminal_regime"
                status="PASS"
                break

        if len(after_ids)<=int(rule["minimum_tessellation_guard_nodes"]):
            stop_reason="minimum_tessellation_guard_reached_without_scientific_stop"
            break

        if (step+1)%checkpoint_every==0:
            ledger.checkpoint(step+1,labels,lineage.active_frame(),{
                "next_step":int(step+1),
                "total_merges":int(total_merges),
                "evaluation_index":int(eval_idx),
                "next_evaluation_removed_fraction":float(next_eval),
            })
    else:
        stop_reason="microstep_safety_ceiling"

    final_ids,final_mass=mass_counts(labels,n0)
    assert_lineage_matches_labels(lineage,labels,n0)
    ledger.checkpoint(
        min(step+1,max_steps),labels,lineage.active_frame(),{
            "next_step":int(min(step+1,max_steps)),
            "total_merges":int(total_merges),
            "evaluation_index":int(eval_idx),
            "next_evaluation_removed_fraction":float(next_eval),
            "terminal":True,
            "stop_reason":stop_reason,
        }
    )

    summary={
        "sample":sample,
        "level0_cells":int(n0),
        "final_nodes":int(len(final_ids)),
        "total_merges":int(total_merges),
        "removed_fraction":float((n0-len(final_ids))/n0),
        "microsteps":int(min(step+1,max_steps)),
        "scientific_evaluations":int(eval_idx),
        "stop_reason":stop_reason,
        "terminal_rule_success":bool(status=="PASS"),
        "exact_partition_invariant":True,
        "exact_mass_invariant":True,
        "lineage_mass_invariant":True,
        "initial_components_preserved":True,
        "frozen_terminal_rule_sha256":EXPECTED_TERMINAL_RULE_SHA256,
        "elapsed_seconds":float(time.time()-t0),
        "status":status,
    }
    writejson(summary,sample_out/"frozen_production_certificate.json")
    return summary
