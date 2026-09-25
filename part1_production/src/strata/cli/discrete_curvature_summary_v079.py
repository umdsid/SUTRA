from __future__ import annotations

import argparse,json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from strata_hierarchy.v071.resource_intake import sha256_file
from strata_hierarchy.v079.curvature_summary import (
    resolve_xy_columns,
    attach_spatial_geometry,
    attach_holonomy_density,
    attach_components,
    summarize_loops,
    cell_aggregate,
    component_aggregate,
    certify_summary,
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


def one(project,sample):
    l0=project/"results"/"hierarchy_level0_v070"/sample
    v77=project/"results"/"hierarchy_v077_discrete_transport"/sample
    v78=project/"results"/"hierarchy_v078_transport_holonomy"/sample

    cells=pd.read_parquet(require(l0/"cells.parquet"))
    loops=pd.read_parquet(require(v78/"all_local_holonomy_loops.parquet"))
    nodes=pd.read_parquet(require(v77/"transport_node_registry.parquet"))

    if len(nodes)!=len(cells):
        raise RuntimeError(
            f"{sample}: transport-node/cell count mismatch "
            f"{len(nodes)} != {len(cells)}"
        )

    xcol,ycol=resolve_xy_columns(cells)

    x=attach_spatial_geometry(loops,cells,xcol,ycol)
    x=attach_holonomy_density(x)
    x=attach_components(
        x,nodes.transport_component.to_numpy(np.int64)
    )

    summary=summarize_loops(x)
    cert=certify_summary(x)
    cell=cell_aggregate(x,len(cells))
    comp=component_aggregate(x)

    # Preserve coordinates for direct spatial plotting.
    cell[xcol]=cells[xcol].to_numpy()
    cell[ycol]=cells[ycol].to_numpy()

    out=project/"results"/"hierarchy_v079_loop_normalized_geometry"/sample
    out.mkdir(parents=True,exist_ok=True)

    writepq(x,out/"loop_holonomy_density.parquet")
    writepq(cell,out/"cell_holonomy_summary.parquet")
    writepq(comp,out/"component_holonomy_summary.parquet")

    figloops=x[
        [
            c for c in [
                "loop_type","loop_index","loop_nodes",
                "n_directional_edges","fully_resolved",
                "spatial_area","spatial_perimeter","spatial_compactness",
                "spectral_angle_rms","spectral_angle_max",
                "holonomy_density_rms","holonomy_density_max",
                "transport_component",
            ] if c in x.columns
        ]
    ]
    writepq(figloops,out/"figure_ready_loop_geometry.parquet")
    writepq(cell,out/"figure_ready_cell_geometry.parquet")

    report={
        "sample":sample,
        "spatial_coordinate_columns":{"x":xcol,"y":ycol},
        "loop_summary":summary,
        "certificate":cert,
        "n_cells_with_directional_loop":int(
            (cell.incident_directional_loops>0).sum()
        ),
        "fraction_cells_with_directional_loop":float(
            (cell.incident_directional_loops>0).mean()
        ),
        "n_transport_components_with_directional_loop":int(len(comp)),
        "policy":{
            "public_observable_name":"holonomy density",
            "internal_descriptor":"loop-normalized discrete curvature-like summary",
            "continuum_curvature_tensor_claimed":False,
            "gaussian_curvature_claimed":False,
            "ricci_curvature_claimed":False,
            "scalar_curvature_claimed":False,
            "normalization":"certified spectral holonomy angle / physical loop polygon area",
            "degenerate_loops_divided_by_area":False,
            "overlapping_loops_treated_as_independent_replicates":False,
            "cell_and_component_aggregates_exported":True,
        },
        "status":"PASS" if cert["certificate_pass"] else "HOLD",
    }
    (out/"loop_normalized_geometry_certificate.json").write_text(
        json.dumps(report,indent=2)+"\n"
    )
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    a=ap.parse_args()

    project=Path(a.project_root).resolve()
    src=require(
        project/"results"/"hierarchy_v078_transport_holonomy"/
        "transport_holonomy_global_certificate.json"
    )
    d=json.loads(src.read_text())
    if d.get("READY_FOR_DISCRETE_CURVATURE_SUMMARY") is not True:
        raise SystemExit("ERROR: v0.7.8 holonomy is not certified")

    print("STRATA 0.7.9 | Loop-normalized transport geometry")
    print("Public observable: holonomy density = certified loop angle / spatial area.")
    print("No smooth curvature tensor is inferred.")
    print("Cell and transport-component summaries are exported.\n")

    reports=[]
    for s in SAMPLES:
        r=one(project,s); reports.append(r)
        q=r["loop_summary"]
        print(
            f"[DONE] {s}: directional_loops={q['n_directional_density_loops']:,} "
            f"cells={100*r['fraction_cells_with_directional_loop']:.2f}% "
            f"components={r['n_transport_components_with_directional_loop']:,} "
            f"density_q95={q['density_rms_q95']:.4g} "
            f"aggregate={q['aggregate_angle_per_area']:.4g} "
            f"{r['status']}"
        )

    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v079_loop_normalized_geometry"
    cert={
        "strata_version":"0.7.9",
        "stage":"loop-normalized transport geometry",
        "source_v078_certificate_sha256":sha256_file(src),
        "mathematical_policy":{
            "primary_observable":"holonomy density",
            "definition":"spectral holonomy angle RMS / physical loop polygon area",
            "continuum_curvature_tensor_inferred":False,
            "loop_geometry_source":"Level-0 spatial centroids",
            "spatially_degenerate_loops":"density undefined",
            "identity_only_loops":"zero density",
            "cell_aggregation":"incident-loop descriptive burden",
            "component_aggregation":"component-level descriptive summaries",
            "independent_replicate_claim_for_loops":False,
        },
        "sample_reports":reports,
        "LOOP_NORMALIZED_GEOMETRY_GATE":"PASS" if gate else "HOLD",
        "HOLONOMY_DENSITY_CERTIFIED":bool(gate),
        "READY_FOR_SPATIAL_GEOMETRY_STATISTICS":bool(gate),
    }
    p=out/"loop_normalized_geometry_global_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print(
        f"\nLOOP-NORMALIZED GEOMETRY GATE: "
        f"{cert['LOOP_NORMALIZED_GEOMETRY_GATE']}"
    )
    print(
        f"HOLONOMY DENSITY CERTIFIED: "
        f"{cert['HOLONOMY_DENSITY_CERTIFIED']}"
    )
    print(
        "READY FOR SPATIAL GEOMETRY STATISTICS: "
        f"{cert['READY_FOR_SPATIAL_GEOMETRY_STATISTICS']}"
    )
    print(f"Certificate: {p}")


if __name__=="__main__":
    main()
