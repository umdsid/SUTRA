from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from strata.mechanics.inference import MechanicsConfig, select_mechanics_edges, solve_mechanics, attach
from strata.mechanics.stress import derive_stress
from strata.mechanics.validation import validate

def find_cells(root:Path):
    hits=list(root.rglob("*cells.parquet"))+list(root.rglob("*cells.parquet.gz"))
    if not hits: raise FileNotFoundError(f"No cells parquet under {root}")
    return hits[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--min-confidence-class",default="admissible",choices=["marginal","admissible","high_confidence"])
    args=ap.parse_args()
    project=Path(args.project_root).resolve()
    gateb=project/"results"/"tranche2_1_gateB"/"gateB_certificate.json"
    if not gateb.exists(): raise SystemExit(f"Missing Gate B certificate: {gateb}")
    cert=json.loads(gateb.read_text())
    if cert.get("gateB_status")!="PASS": raise SystemExit("Gate B is not PASS; mechanics blocked.")
    cfg=MechanicsConfig(min_confidence_class=args.min_confidence_class)
    outroot=project/"results"/"tranche3_mechanics"; outroot.mkdir(parents=True,exist_ok=True)
    print("STRATA 0.3.0 | Tranche 3 | Certified Mechanical State")
    print(f"Project: {project}")
    print(f"Gate B epsilon: {cert['frozen_contact_epsilon']}")
    print(f"Mechanics confidence floor: {cfg.min_confidence_class}")
    reports=[]
    for k,sc in enumerate(cert["sample_certificates"],1):
        name=sc["sample"]; print(f"[{k}/{len(cert['sample_certificates'])}] {name}")
        ci=pd.read_parquet(project/"results"/"tranche2_1_gateB"/name/"certified_interfaces.parquet")
        topo=pd.read_parquet(project/"results"/"tranche2_topology"/name/"observed_interfaces.parquet")
        topo=topo.copy()
        topo["key"]=topo.apply(lambda r: tuple(sorted((str(r.cell_i),str(r.cell_j)))),axis=1)
        ci=ci.copy()
        ci["key"]=ci.apply(lambda r: tuple(sorted((str(r.cell_i),str(r.cell_j)))),axis=1)
        merged=ci.merge(topo[["key","normal_i_to_j_x","normal_i_to_j_y"]],on="key",how="left").drop(columns=["key"])
        medges=select_mechanics_edges(merged,cfg)
        cells=pd.read_parquet(find_cells(project/"data"/name))
        cells["cell_id"]=cells["cell_id"].astype(str)
        ids=cells["cell_id"].tolist()
        sol=solve_mechanics(medges,ids,cfg)
        me,p=attach(medges,ids,sol)
        stress=derive_stress(cells,me,p)
        rep=validate(me,p,stress,sol)
        rep.update(sample=name,gateB_epsilon=cert["frozen_contact_epsilon"],mechanics_edge_floor=cfg.min_confidence_class)
        out=outroot/name; out.mkdir(parents=True,exist_ok=True)
        me.to_parquet(out/"interface_mechanics.parquet",index=False)
        p.to_parquet(out/"cell_pressure_residual.parquet",index=False)
        stress.to_parquet(out/"cell_stress.parquet",index=False)
        (out/"mechanics_summary.json").write_text(json.dumps(rep,indent=2))
        reports.append(rep)
        print(f"    edges={rep['n_mechanics_edges']:,} cells={rep['n_mechanics_cells']:,} stress_valid={100*rep['stress_valid_fraction']:.2f}% q95_resid={rep['q95_force_balance_residual']:.3e} status={rep['status']}")
    overall={
        "strata_version":"0.3.0","tranche":3,"gate":"C",
        "contract":"certified mechanical state; strain excluded",
        "sample_reports":reports,
        "gateC_status":"PASS" if all(r["status"]=="PASS" for r in reports) else "FAIL",
    }
    (outroot/"gateC_certificate.json").write_text(json.dumps(overall,indent=2))
    print()
    print(f"Gate C: {overall['gateC_status']}")
    print(f"Certificate: {outroot/'gateC_certificate.json'}")

if __name__=="__main__":
    main()
