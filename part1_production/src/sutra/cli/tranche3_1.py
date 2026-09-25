from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

from sutra.io.boundaries import read_boundary_table,polygons_from_boundary_table
from sutra.mechanics.junctions import materialize_interface_geometry,cluster_endpoints
from sutra.mechanics.vertex_solver import VertexMechanicsConfig,solve_vertex_mechanics
from sutra.mechanics.stress_v2 import derive_stress
from sutra.mechanics.gatec import certify


def find_cells(root):
    hits=list(Path(root).rglob("*cells.parquet"))+list(Path(root).rglob("*cells.parquet.gz"))
    if not hits: raise FileNotFoundError(root)
    return hits[0]

def find_boundaries(root):
    hits=list(Path(root).rglob("*cell_boundaries*.parquet"))+list(Path(root).rglob("*cell_boundaries*.parquet.gz"))
    if not hits: raise FileNotFoundError(root)
    return hits[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--junction-merge-tolerance",type=float,default=0.175)
    ap.add_argument("--confidence-floor",default="admissible",choices=["admissible","high_confidence"])
    args=ap.parse_args()

    project=Path(args.project_root).resolve()
    gb=json.loads((project/"results"/"tranche2_1_gateB"/"gateB_certificate.json").read_text())
    if gb.get("gateB_status")!="PASS": raise SystemExit("Gate B not PASS.")
    eps=float(gb["frozen_contact_epsilon"])
    rank={"marginal":0,"admissible":1,"high_confidence":2}
    outroot=project/"results"/"tranche3_1_junction_mechanics"; outroot.mkdir(parents=True,exist_ok=True)

    print("STRATA 0.3.1 | Tranche 3.1 | Junction-based Mechanical State")
    print(f"Gate B epsilon: {eps:.3f}")
    print(f"Junction merge tolerance: {args.junction_merge_tolerance:.3f}")
    reports=[]

    for k,sc in enumerate(gb["sample_certificates"],1):
        name=sc["sample"]; print(f"[{k}/{len(gb['sample_certificates'])}] {name}")
        sample=project/"data"/name
        ci=pd.read_parquet(project/"results"/"tranche2_1_gateB"/name/"certified_interfaces.parquet")
        ci=ci[ci.confidence_class.map(lambda x:rank.get(str(x),-1))>=rank[args.confidence_floor]].reset_index(drop=True)

        bdf=read_boundary_table(find_boundaries(sample))
        polygons,_=polygons_from_boundary_table(bdf)
        geom=materialize_interface_geometry(ci,polygons,eps)
        reconstruction_fraction=len(geom)/max(len(ci),1)
        geom,junctions=cluster_endpoints(geom,args.junction_merge_tolerance)

        cells=pd.read_parquet(find_cells(sample)); cells["cell_id"]=cells.cell_id.astype(str)
        ids=cells.cell_id.tolist()

        sol=solve_vertex_mechanics(geom,ids,len(junctions),VertexMechanicsConfig())
        geom["tension_like"]=sol["tension"]
        pressure=pd.DataFrame({
            "cell_id":ids,
            "pressure_like":sol["pressure"],
        })
        stress=derive_stress(cells,geom,pressure)
        report=certify(geom,junctions,pressure,stress,sol,reconstruction_fraction)
        report.update(sample=name,gateB_epsilon=eps,junction_merge_tolerance=args.junction_merge_tolerance,confidence_floor=args.confidence_floor)

        jr=pd.DataFrame({
            "junction_id":junctions.junction_id,
            "x":junctions.x,"y":junctions.y,
            "n_incident_interfaces":junctions.n_incident_interfaces,
            "force_balance_residual":sol["junction_residual"],
            "normalized_force_balance_residual":sol["normalized_junction_residual"],
        })

        out=outroot/name; out.mkdir(parents=True,exist_ok=True)
        geom.to_parquet(out/"interface_mechanics.parquet",index=False)
        junctions.to_parquet(out/"junction_geometry.parquet",index=False)
        jr.to_parquet(out/"junction_residuals.parquet",index=False)
        pressure.to_parquet(out/"cell_pressure.parquet",index=False)
        stress.to_parquet(out/"cell_stress.parquet",index=False)
        (out/"mechanics_summary.json").write_text(json.dumps(report,indent=2))
        reports.append(report)

        print(
            f"    reconstructed={100*report['reconstruction_fraction']:.2f}% "
            f"edges={report['n_mechanics_edges']:,} junctions={report['n_junctions']:,} "
            f"tau_sd={report['std_tension']:.3e} p_sd={report['std_pressure']:.3e} "
            f"q95_R={report['q95_normalized_junction_residual']:.3f} "
            f"iters={report['solver_iterations']} status={report['status']}"
        )

    overall={
        "strata_version":"0.3.1","tranche":"3.1","gate":"C",
        "contract":"junction-based relative mechanical state; strain excluded",
        "sample_reports":reports,
        "gateC_status":"PASS" if all(r["status"]=="PASS" for r in reports) else "FAIL",
        "supersedes":"Tranche 3 v0.3.0 false-positive certificate",
    }
    (outroot/"gateC_certificate.json").write_text(json.dumps(overall,indent=2))
    print()
    print(f"Gate C: {overall['gateC_status']}")
    if overall["gateC_status"]!="PASS":
        print("Mechanics is NOT frozen; inspect junction/residual diagnostics.")
    print(f"Certificate: {outroot/'gateC_certificate.json'}")

if __name__=="__main__":
    main()
