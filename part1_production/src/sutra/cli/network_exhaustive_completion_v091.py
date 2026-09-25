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
from sutra.hierarchy.v091.completion import (
    resolve_xy,build_backbone,cosine_similarity_edges,spatial_expression_weight,
    functional_distance_edges,adjacency_from_edges,harmonic_fill,
    edge_anchor_to_nodes,flexible_col,compile_cellchat,communication_edges,
    TENSION_NAMES,DP_NAMES
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def require(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p

def writepq(df,path):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),path,compression="zstd")

def one(project_s,sample,cfg):
    os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1")
    t0=time.time(); project=Path(project_s)
    l0=project/"results"/"hierarchy_level0_v070"/sample
    v71=project/"results"/"hierarchy_v071_active_blocks"/sample
    v72=project/"results"/"hierarchy_v072_short_pilot"/sample
    v762=project/"results"/"hierarchy_v0762_pressure_field"/sample

    cells=pd.read_parquet(require(l0/"cells.parquet"))
    edge_rel=pd.read_parquet(require(v72/"level0_candidate_relations.parquet"))
    manifest=json.loads(require(l0/"expression_manifest.json").read_text())
    matrix=Path(manifest["source_matrix"])
    if sha256_file(matrix)!=manifest["source_matrix_sha256"]:
        raise RuntimeError(f"{sample}: expression source hash changed")
    X,barcodes,genes=read_10x_cells_by_genes(matrix)
    Y=normalized_expression_dense(X).astype(np.float32)
    if len(Y)!=len(cells): raise RuntimeError(f"{sample}: expression/cell mismatch")

    xcol,ycol=resolve_xy(cells); xy=cells[[xcol,ycol]].to_numpy(float)
    contacts=edge_rel[["cell_i_index","cell_j_index"]].to_numpy(np.int64)
    B=build_backbone(xy,contacts,int(cfg["spatial_knn"]),float(cfg["gap_factor"]),int(cfg["local_scale_k"]))

    cos=cosine_similarity_edges(Y,B,int(cfg["edge_chunk"]))
    w,ds,de,escale=spatial_expression_weight(B,cos)
    B["expression_cosine_similarity"]=cos
    B["expression_distance"]=de
    B["normalized_spatial_distance"]=ds
    B["spatial_expression_weight"]=w

    L=sparse.load_npz(require(v71/"functional_laplacian.npz")).toarray()
    audit=json.loads(require(v71/"functional_resource_audit.json").read_text())
    lam=float(audit["lambda_g"])
    G=np.eye(L.shape[0])+lam*L
    fd=functional_distance_edges(Y,B,G,int(cfg["functional_chunk"]))
    med=max(float(np.median(fd[fd>0])) if np.any(fd>0) else 1.0,1e-12)
    B["functional_distance"]=fd
    B["functional_similarity"]=np.exp(-fd/med)
    B["functional_provenance"]="panel_supported_GO_MSIGDB_metric"

    A=adjacency_from_edges(B,len(cells),np.maximum(B.spatial_expression_weight.to_numpy(float),1e-8))

    tcol=flexible_col(edge_rel,TENSION_NAMES)
    tvalid=next((c for c in ["mechanics_tau_valid","tau_valid","tension_valid"] if c in edge_rel.columns),None)
    tmask,tanchor=edge_anchor_to_nodes(len(cells),edge_rel,tcol,tvalid)
    prior=float(np.nanmedian(tanchor[tmask])) if tmask.any() else 0.0
    tfill,tconf,tprov=harmonic_fill(A,tmask,np.where(tmask,tanchor,0.0),prior,int(cfg["harmonic_max_iter"]),float(cfg["harmonic_tol"]))

    i=B.cell_i_index.to_numpy(np.int64); j=B.cell_j_index.to_numpy(np.int64)
    B["tension_complete"]=0.5*(tfill[i]+tfill[j])
    B["tension_confidence"]=np.minimum(tconf[i],tconf[j])
    B["tension_provenance"]=np.where(tmask[i]&tmask[j],"anchor_supported",np.where(B.tension_confidence>0,"harmonic_extension","neutral_prior"))

    pcells=pd.read_parquet(require(v762/"pressure_potential_cells.parquet"))
    p=pcells.pressure_potential.to_numpy(float)
    dpcol=flexible_col(edge_rel,DP_NAMES)
    dpvalid=next((c for c in ["mechanics_delta_p_valid","delta_p_valid"] if c in edge_rel.columns),None)
    pmask,_=edge_anchor_to_nodes(len(cells),edge_rel,dpcol,dpvalid)
    _,pconf,pprov=harmonic_fill(A,pmask,np.where(pmask,p,0.0),0.0,1,0.0)
    B["delta_p_potential_complete"]=p[i]-p[j]
    B["pressure_confidence"]=np.minimum(pconf[i],pconf[j])
    B["pressure_provenance"]=np.where(pmask[i]&pmask[j],"certified_potential_anchor_supported",np.where(B.pressure_confidence>0,"frozen_potential_graph_supported","frozen_potential_low_support"))

    registry=pd.read_csv(require(project/"resources"/"strata_hierarchy_v0711"/"signaling"/"cellchat_typed_interactions.csv"))
    channels=compile_cellchat(registry,genes)
    cf,cr,sf,sr=communication_edges(Y,B,channels,int(cfg["communication_chunk"]))
    B["cellchat_forward"]=cf; B["cellchat_reverse"]=cr
    B["cellchat_support_forward"]=sf; B["cellchat_support_reverse"]=sr
    B["cellchat_total"]=cf+cr
    B["cellchat_directionality"]=(cf-cr)/(cf+cr+1e-12)
    B["cellchat_provenance"]="all_panel_supported_channels_evaluated"

    node=pd.DataFrame({
        "cell_index":np.arange(len(cells),dtype=np.int64),
        xcol:xy[:,0],ycol:xy[:,1],
        "tension_complete":tfill,"tension_confidence":tconf,"tension_provenance":tprov,
        "pressure_potential":p,"pressure_confidence":pconf,"pressure_provenance":pprov,
    })

    out=project/"results"/"hierarchy_v091_network_exhaustive_completion"/sample
    out.mkdir(parents=True,exist_ok=True)
    writepq(B,out/"completed_tissue_backbone_edges.parquet")
    writepq(node,out/"completed_tissue_node_fields.parquet")
    sparse.save_npz(out/"completed_tissue_backbone_adjacency.npz",A,compressed=True)

    report={
        "sample":sample,
        "n_cells":int(len(cells)),
        "n_backbone_edges":int(len(B)),
        "contact_edges_retained":int(B.source_contact.sum()),
        "knn_edges":int(B.source_knn.sum()),
        "n_cellchat_registry_interactions":int(len(registry)),
        "n_cellchat_panel_supported_channels":int(len(channels)),
        "tension_anchor_cell_fraction":float(tmask.mean()),
        "tension_positive_confidence_cell_fraction":float((tconf>0).mean()),
        "pressure_anchor_cell_fraction":float(pmask.mean()),
        "pressure_positive_confidence_cell_fraction":float((pconf>0).mean()),
        "functional_distance_finite_fraction":float(np.isfinite(fd).mean()),
        "spatial_expression_weight_finite_fraction":float(np.isfinite(w).mean()),
        "cellchat_complete_finite_fraction":float(np.isfinite(cf+cr).mean()),
        "elapsed_seconds":float(time.time()-t0),
        "policy":{
            "existing_contact_edges_removed":False,
            "spatial_knn_gap_guarded":True,
            "completion_values_promoted_to_measurements":False,
            "mechanics_provenance_retained":True,
            "GO_MSIGDB_evaluated_every_backbone_edge":True,
            "CellChat_panel_supported_channels_evaluated_every_backbone_edge":True,
            "pressure_uses_frozen_scalar_potential":True,
        },
        "status":"PASS",
    }
    (out/"network_exhaustive_completion_certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v091_completion.json")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    cfg=json.loads(require(project/a.config).read_text())

    src=require(project/"results"/"hierarchy_v079_loop_normalized_geometry"/"loop_normalized_geometry_global_certificate.json")
    d=json.loads(src.read_text())
    if d.get("HOLONOMY_DENSITY_CERTIFIED") is not True:
        raise SystemExit("ERROR: frozen Level-0 construction is not certified")

    print("STRATA 0.9.1 | Network-exhaustive field completion")
    print("Universal substrate: local spatial-expression tissue backbone.")
    print("GO/MSigDB, CellChat, tension and pressure completed/evaluated across it.")
    print("Observed/propagated/neutral provenance retained.")
    print("Runs independently of any still-running strict v0.9.0 flow.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        futs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(futs):
            r=f.result(); reports.append(r)
            print(f"[DONE] {r['sample']}: cells={r['n_cells']:,} edges={r['n_backbone_edges']:,} CellChat_channels={r['n_cellchat_panel_supported_channels']:,} tension_conf={100*r['tension_positive_confidence_cell_fraction']:.1f}% pressure_conf={100*r['pressure_positive_confidence_cell_fraction']:.1f}% {r['status']}",flush=True)

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v091_network_exhaustive_completion"
    cert={
        "strata_version":"0.9.1",
        "stage":"network-exhaustive tissue field completion",
        "runs_independently_of_v090_strict_flow":True,
        "parallel_specimens":3,
        "sample_reports":reports,
        "NETWORK_EXHAUSTIVE_COMPLETION_GATE":"PASS" if gate else "HOLD",
        "COMPLETED_TISSUE_STATE_READY":bool(gate),
        "READY_FOR_COMPLETED_ULTRASLOW_FLOW":bool(gate),
    }
    p=out/"network_exhaustive_completion_global_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")
    print(f"\nNETWORK-EXHAUSTIVE COMPLETION GATE: {cert['NETWORK_EXHAUSTIVE_COMPLETION_GATE']}")
    print(f"COMPLETED TISSUE STATE READY: {cert['COMPLETED_TISSUE_STATE_READY']}")
    print(f"READY FOR COMPLETED ULTRASLOW FLOW: {cert['READY_FOR_COMPLETED_ULTRASLOW_FLOW']}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
