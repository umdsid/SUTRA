from __future__ import annotations
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd

from sutra.mechanics.vertex_exact import (
    VertexDomainConfig,build_cell_vertex_cycles,build_exact_vertex_system
)
from sutra.mechanics.vertex_exact_solver import solve_exact_vertex
from sutra.mechanics.vertex_exact_audit import summarize_vertex_mechanics
from sutra.mechanics.stress_v2 import derive_stress


def _run_sample(payload):
    project_s,rep=payload
    project=Path(project_s)
    name=rep["sample"]
    src=project/"results"/"tranche3_2_gateC"/name

    edges=pd.read_parquet(src/"interface_mechanics.parquet")
    junctions=pd.read_parquet(src/"junction_geometry.parquet")
    old_pressure=pd.read_parquet(src/"cell_pressure.parquet")

    cells=pd.read_parquet(list((project/"data"/name).rglob("*cells.parquet"))[0])
    cells["cell_id"]=cells["cell_id"].astype(str)

    cycles,gdiag=build_cell_vertex_cycles(
        edges,junctions,cells,
        VertexDomainConfig(min_cell_vertices=3,max_cell_area_rel_error=1.0)
    )

    A,domain_edges,domain_cells,used_j=build_exact_vertex_system(
        edges,junctions,cycles
    )
    m=len(domain_edges);n=len(domain_cells)
    if m==0 or n==0 or A.shape[0]==0:
        raise RuntimeError("empty exact-vertex mechanics domain")

    # Warm start from the existing Gate-C field only as an optimizer initial point.
    edge_key=lambda a,b: tuple(sorted((str(a),str(b))))
    old_tau={
        edge_key(r.cell_i,r.cell_j):float(r.tension_like)
        for r in edges.itertuples(index=False)
    }
    init_tau=np.array([
        old_tau.get(edge_key(r.cell_i,r.cell_j),1.0)
        for r in domain_edges.itertuples(index=False)
    ],float)
    pmap=dict(zip(old_pressure.cell_id.astype(str),old_pressure.pressure_like.astype(float)))
    init_p=np.array([pmap.get(cid,0.0) for cid in domain_cells],float)

    sol=solve_exact_vertex(A,m,n,init_tau,init_p)

    domain_edges=domain_edges.copy()
    domain_edges["tension_like"]=sol["tension"]
    pressure=pd.DataFrame({"cell_id":domain_cells,"pressure_like":sol["pressure"]})

    # Stress only on exact-vertex mechanics domain.
    cells_domain=cells[cells.cell_id.isin(set(domain_cells))].copy()
    stress=derive_stress(cells_domain,domain_edges,pressure)

    jtab=junctions.set_index("junction_id")
    residuals=pd.DataFrame({
        "junction_id":used_j,
        "x":[float(jtab.loc[j,"x"]) for j in used_j],
        "y":[float(jtab.loc[j,"y"]) for j in used_j],
        "vertex_force_residual":sol["vertex_residual"],
    })

    summary=summarize_vertex_mechanics(
        sol,gdiag,len(cells),len(domain_cells),len(edges),len(domain_edges)
    )
    summary["sample"]=name

    out=project/"results"/"tranche3_6_vertex_mechanics"/name
    out.mkdir(parents=True,exist_ok=True)
    cycles.to_parquet(out/"cell_vertex_cycles.parquet",index=False)
    gdiag.to_parquet(out/"cell_vertex_geometry_diagnostics.parquet",index=False)
    domain_edges.to_parquet(out/"interface_tension_vertex.parquet",index=False)
    pressure.to_parquet(out/"cell_pressure_vertex.parquet",index=False)
    stress.to_parquet(out/"cell_stress_vertex.parquet",index=False)
    residuals.to_parquet(out/"vertex_residuals.parquet",index=False)
    (out/"vertex_mechanics_summary.json").write_text(json.dumps(summary,indent=2))
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--sample-workers",type=int,default=3)
    args=ap.parse_args()
    project=Path(args.project_root).resolve()

    gatec=json.loads((project/"results"/"tranche3_2_gateC"/"gateC_certificate.json").read_text())
    if gatec.get("gateC_status")!="PASS":
        raise SystemExit("Existing Gate C certificate must be PASS to supply certified junction geometry.")

    reps=gatec["sample_reports"]
    workers=min(max(1,args.sample_workers),len(reps))
    outroot=project/"results"/"tranche3_6_vertex_mechanics"
    outroot.mkdir(parents=True,exist_ok=True)

    print("STRATA 0.4.2 | Tranche 3.6 | Exact polygonal vertex mechanics")
    print(f"Running {len(reps)} specimens on {workers} workers.",flush=True)
    print("Pressure coefficients: exact polygon-area derivatives.",flush=True)
    print("Tension coefficients: exact interface-length derivatives.",flush=True)

    reports=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(_run_sample,(str(project),r)):r["sample"] for r in reps}
        for fut in as_completed(futs):
            name=futs[fut]
            try:
                r=fut.result()
                print(
                    f"[DONE] {name}: "
                    f"cells={100*r['mechanics_domain_cell_fraction']:.1f}% "
                    f"edges={100*r['mechanics_domain_edge_fraction']:.1f}% "
                    f"Neff={100*r['tension_neff_fraction']:.2f}% "
                    f"top1={100*r['tension_top1_mass']:.1f}% "
                    f"q95R={r['q95_vertex_residual']:.3e} "
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
        "strata_version":"0.4.2",
        "tranche":"3.6",
        "contract":"exact polygonal vertex mechanics using analytic area and length derivatives; strain excluded",
        "sample_reports":reports,
        "run_status":"PASS" if all("reason" not in r for r in reports) else "FAIL",
        "note":"This tranche is a model replacement experiment. Mechanics is not frozen until identifiability is audited."
    }
    (outroot/"tranche3_6_certificate.json").write_text(json.dumps(overall,indent=2))
    print()
    print(f"Tranche 3.6 run: {overall['run_status']}")
    print(f"Certificate: {outroot/'tranche3_6_certificate.json'}")


if __name__=="__main__":
    main()
