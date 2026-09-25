from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

from strata.io.boundaries import read_boundary_table,polygons_from_boundary_table
from strata.mechanics.junctions import materialize_interface_geometry
from strata.mechanics.junction_calibration import calibrate_junction_tolerance
from strata.mechanics.vertex_solver_v2 import solve,SolverConfig
from strata.mechanics.stress_v2 import derive_stress
from strata.mechanics.gatec_v2 import certify

def find_one(root,pattern):
    hits=list(Path(root).rglob(pattern))
    if not hits: raise FileNotFoundError(f"{pattern} under {root}")
    return hits[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--confidence-floor",default="admissible",choices=["admissible","high_confidence"])
    args=ap.parse_args()
    project=Path(args.project_root).resolve()
    gb=json.loads((project/"results"/"tranche2_1_gateB"/"gateB_certificate.json").read_text())
    if gb.get("gateB_status")!="PASS": raise SystemExit("Gate B not PASS.")
    eps=float(gb["frozen_contact_epsilon"])
    rank={"marginal":0,"admissible":1,"high_confidence":2}
    outroot=project/"results"/"tranche3_2_gateC";outroot.mkdir(parents=True,exist_ok=True)

    print("STRATA 0.3.3 | Tranche 3.2 | Gate C mechanics calibration")
    print(f"Gate B epsilon: {eps:.3f}")
    reports=[]

    for k,sc in enumerate(gb["sample_certificates"],1):
        name=sc["sample"];print(f"[{k}/{len(gb['sample_certificates'])}] {name}")
        sample=project/"data"/name
        ci=pd.read_parquet(project/"results"/"tranche2_1_gateB"/name/"certified_interfaces.parquet")
        ci=ci[ci.confidence_class.map(lambda x:rank.get(str(x),-1))>=rank[args.confidence_floor]].reset_index(drop=True)

        bdf=read_boundary_table(find_one(sample,"*cell_boundaries*.parquet"))
        polys,_=polygons_from_boundary_table(bdf)
        raw=materialize_interface_geometry(ci,polys,eps)
        reconstruction=len(raw)/max(len(ci),1)

        geom,junctions,sweep,jtol=calibrate_junction_tolerance(raw,eps)

        cells=pd.read_parquet(find_one(sample,"*cells.parquet"))
        cells["cell_id"]=cells.cell_id.astype(str)
        ids=cells.cell_id.tolist()

        sol=solve(geom,ids,len(junctions),SolverConfig())
        geom=geom.copy();geom["tension_like"]=sol["tension"]
        pressure=pd.DataFrame({"cell_id":ids,"pressure_like":sol["pressure"]})
        stress=derive_stress(cells,geom,pressure)
        rep=certify(geom,junctions,pressure,stress,sol,reconstruction)
        rep.update(
            sample=name,gateB_epsilon=eps,
            calibrated_junction_tolerance=float(jtol),
            confidence_floor=args.confidence_floor
        )

        jr=pd.DataFrame({
            "junction_id":junctions.junction_id,
            "x":junctions.x,"y":junctions.y,
            "n_incident_interfaces":junctions.n_incident_interfaces,
            "fitted_residual":sol["junction_residual"],
            "baseline_residual":sol["baseline_junction_residual"],
        })

        out=outroot/name;out.mkdir(parents=True,exist_ok=True)
        raw.to_parquet(out/"raw_interface_geometry.parquet",index=False)
        geom.to_parquet(out/"interface_mechanics.parquet",index=False)
        junctions.to_parquet(out/"junction_geometry.parquet",index=False)
        jr.to_parquet(out/"junction_residuals.parquet",index=False)
        pressure.to_parquet(out/"cell_pressure.parquet",index=False)
        stress.to_parquet(out/"cell_stress.parquet",index=False)
        sweep.to_csv(out/"junction_tolerance_sweep.csv",index=False)
        (out/"mechanics_summary.json").write_text(json.dumps(rep,indent=2))
        reports.append(rep)

        print(
            f"    jtol={jtol:.3f} tri={100*rep['junction_degree_ge3_fraction']:.2f}% "
            f"tau_mean={rep['mean_tension']:.6f} tau_sd={rep['std_tension']:.3e} "
            f"p_sd={rep['std_pressure']:.3e} "
            f"median_R={rep['median_junction_residual']:.3f}/{rep['baseline_median_junction_residual']:.3f} "
            f"q95_R={rep['q95_junction_residual']:.3f}/{rep['baseline_q95_junction_residual']:.3f} "
            f"improved={100*rep['fraction_improved']:.1f}% status={rep['status']}"
        )

    overall={
        "strata_version":"0.3.3",
        "tranche":"3.2",
        "gate":"C",
        "contract":"data-calibrated junction mechanics; exact scale/gauge; strain excluded",
        "sample_reports":reports,
        "gateC_status":"PASS" if all(r["status"]=="PASS" for r in reports) else "FAIL",
        "supersedes":["0.3.0","0.3.1","0.3.2"],
    }
    (outroot/"gateC_certificate.json").write_text(json.dumps(overall,indent=2))
    print()
    print(f"Gate C: {overall['gateC_status']}")
    print(f"Certificate: {outroot/'gateC_certificate.json'}")

if __name__=="__main__":
    main()
