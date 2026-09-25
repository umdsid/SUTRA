from __future__ import annotations
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd

from sutra.io.boundaries import read_boundary_table,polygons_from_boundary_table
from sutra.mechanics.junctions import reconstruct_interface_arc
from sutra.mechanics.vertex_exact import (
    VertexDomainConfig,build_cell_vertex_cycles,build_exact_vertex_system
)
from sutra.mechanics.curvature_constraints import (
    CurvatureConfig,signed_curvature_from_arc,build_young_laplace_matrix
)
from sutra.mechanics.curvature_solver import solve_curvature_constrained
from sutra.mechanics.vertex_exact_audit import neff_fraction,top_mass
from sutra.mechanics.stress_v2 import derive_stress


def _find(root,pattern):
    hits=list(Path(root).rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"{pattern} under {root}")
    return hits[0]


def _run_sample(payload):
    project_s,rep=payload
    project=Path(project_s)
    name=rep["sample"]

    gateb=json.loads((project/"results"/"tranche2_1_gateB"/"gateB_certificate.json").read_text())
    eps=float(gateb["frozen_contact_epsilon"])

    src=project/"results"/"tranche3_2_gateC"/name
    gate_edges=pd.read_parquet(src/"interface_mechanics.parquet")
    junctions=pd.read_parquet(src/"junction_geometry.parquet")

    cells=pd.read_parquet(_find(project/"data"/name,"*cells.parquet"))
    cells["cell_id"]=cells.cell_id.astype(str)

    bdf=read_boundary_table(_find(project/"data"/name,"*cell_boundaries*.parquet"))
    polys,_=polygons_from_boundary_table(bdf)

    cycles,gdiag=build_cell_vertex_cycles(
        gate_edges,junctions,cells,
        VertexDomainConfig(min_cell_vertices=3,max_cell_area_rel_error=1.0)
    )
    A,edges,domain_cells,used_j=build_exact_vertex_system(
        gate_edges,junctions,cycles
    )
    m=len(edges);n=len(domain_cells)
    if m==0 or n==0:
        raise RuntimeError("empty vertex domain")

    # Materialize measured arcs and fit reliable curvature.
    crow=[]
    cfg=CurvatureConfig()
    for r in edges.itertuples(index=False):
        a,b=str(r.cell_i),str(r.cell_j)
        gi,gj=polys.get(a),polys.get(b)
        if gi is None or gj is None:
            continue
        arc=reconstruct_interface_arc(gi,gj,eps)
        if arc is None:
            out={"curvature_valid":False,"reason":"arc_reconstruction_failed"}
        else:
            nij=[float(r.normal_i_to_j_x),float(r.normal_i_to_j_y)]
            out=signed_curvature_from_arc(arc,nij,cfg)
        out.update({"cell_i":a,"cell_j":b})
        crow.append(out)
    ctab=pd.DataFrame(crow)

    Y,ymeta=build_young_laplace_matrix(edges,domain_cells,ctab)

    # Warm start from exact-vertex 3.6 if available; otherwise Gate-C.
    v36=project/"results"/"tranche3_6_vertex_mechanics"/name
    if (v36/"interface_tension_vertex.parquet").exists():
        olde=pd.read_parquet(v36/"interface_tension_vertex.parquet")
        oldp=pd.read_parquet(v36/"cell_pressure_vertex.parquet")
    else:
        olde=gate_edges
        oldp=pd.read_parquet(src/"cell_pressure.parquet")

    key=lambda a,b:tuple(sorted((str(a),str(b))))
    tmap={key(r.cell_i,r.cell_j):float(r.tension_like) for r in olde.itertuples(index=False)}
    pmap=dict(zip(oldp.cell_id.astype(str),oldp.pressure_like.astype(float)))
    init_tau=np.array([tmap.get(key(r.cell_i,r.cell_j),1.0) for r in edges.itertuples(index=False)],float)
    init_p=np.array([pmap.get(c,0.0) for c in domain_cells],float)

    sol=solve_curvature_constrained(A,Y,m,n,init_tau,init_p)

    edges=edges.copy()
    edges["tension_like"]=sol["tension"]
    pressure=pd.DataFrame({"cell_id":domain_cells,"pressure_like":sol["pressure"]})
    stress=derive_stress(cells[cells.cell_id.isin(set(domain_cells))],edges,pressure)

    valid_curv=int(ctab.curvature_valid.sum()) if len(ctab) and "curvature_valid" in ctab else 0
    curv_frac=valid_curv/max(len(edges),1)

    report={
        "sample":name,
        "n_domain_edges":m,
        "n_domain_cells":n,
        "n_reliable_curvature_edges":valid_curv,
        "reliable_curvature_fraction":float(curv_frac),
        "mean_tension":float(np.mean(sol["tension"])),
        "std_tension":float(np.std(sol["tension"])),
        "tension_neff_fraction":float(neff_fraction(sol["tension"])),
        "tension_top1_mass":float(top_mass(sol["tension"],0.01)),
        "median_vertex_residual":float(np.median(sol["vertex_residual"])),
        "q95_vertex_residual":float(np.quantile(sol["vertex_residual"],0.95)),
        "median_young_laplace_residual":float(np.median(np.abs(sol["young_laplace_residual"]))) if len(sol["young_laplace_residual"]) else None,
        "q95_young_laplace_residual":float(np.quantile(np.abs(sol["young_laplace_residual"]),0.95)) if len(sol["young_laplace_residual"]) else None,
        "stress_valid_fraction":float(stress.stress_valid.mean()),
        "solver_success":bool(sol["success"]),
        "solver_iterations":int(sol["iterations"]),
        "solver_optimality":float(sol["optimality"]),
    }

    out=project/"results"/"tranche3_7_curvature_mechanics"/name
    out.mkdir(parents=True,exist_ok=True)
    ctab.to_parquet(out/"interface_curvature.parquet",index=False)
    ymeta.to_parquet(out/"young_laplace_constraints.parquet",index=False)
    edges.to_parquet(out/"interface_tension_curvature.parquet",index=False)
    pressure.to_parquet(out/"cell_pressure_curvature.parquet",index=False)
    stress.to_parquet(out/"cell_stress_curvature.parquet",index=False)
    gdiag.to_parquet(out/"cell_vertex_geometry_diagnostics.parquet",index=False)
    (out/"curvature_mechanics_summary.json").write_text(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--sample-workers",type=int,default=3)
    args=ap.parse_args()
    project=Path(args.project_root).resolve()

    gatec=json.loads((project/"results"/"tranche3_2_gateC"/"gateC_certificate.json").read_text())
    reps=gatec["sample_reports"]
    workers=min(max(1,args.sample_workers),len(reps))

    outroot=project/"results"/"tranche3_7_curvature_mechanics"
    outroot.mkdir(parents=True,exist_ok=True)

    print("STRATA 0.4.3 | Tranche 3.7 | Curvature-constrained vertex mechanics")
    print(f"Running {len(reps)} specimens on {workers} workers.",flush=True)
    print("Young-Laplace constraints are used only on reliable measured arcs.",flush=True)

    reports=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(_run_sample,(str(project),r)):r["sample"] for r in reps}
        for fut in as_completed(futs):
            name=futs[fut]
            try:
                r=fut.result()
                print(
                    f"[DONE] {name}: "
                    f"curv={100*r['reliable_curvature_fraction']:.1f}% "
                    f"Neff={100*r['tension_neff_fraction']:.2f}% "
                    f"top1={100*r['tension_top1_mass']:.1f}% "
                    f"q95V={r['q95_vertex_residual']:.3e} "
                    f"q95YL={r['q95_young_laplace_residual'] if r['q95_young_laplace_residual'] is not None else float('nan'):.3e} "
                    f"iters={r['solver_iterations']} success={r['solver_success']}",
                    flush=True
                )
            except Exception as e:
                r={"sample":name,"status":"FAIL","reason":f"{type(e).__name__}: {e}"}
                print(f"[DONE] {name}: ERROR {r['reason']}",flush=True)
            reports.append(r)

    order={r["sample"]:i for i,r in enumerate(reps)}
    reports.sort(key=lambda r:order.get(r["sample"],999))

    overall={
        "strata_version":"0.4.3",
        "tranche":"3.7",
        "contract":"reliable-curvature Young-Laplace constraints plus exact polygonal vertex equilibrium; strain excluded",
        "sample_reports":reports,
        "run_status":"PASS" if all("reason" not in r for r in reports) else "FAIL",
        "terminal_decision_rule":"If all three specimens retain pathological top-1%-tension concentration despite reliable curvature constraints, inferred pressure/tension will be dropped from the paper."
    }
    (outroot/"tranche3_7_certificate.json").write_text(json.dumps(overall,indent=2))
    print()
    print(f"Tranche 3.7 run: {overall['run_status']}")
    print(f"Certificate: {outroot/'tranche3_7_certificate.json'}")


if __name__=="__main__":
    main()
