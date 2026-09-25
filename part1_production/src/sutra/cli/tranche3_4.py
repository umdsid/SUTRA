from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

from sutra.mechanics.reg_calibration import calibrate_lambda
from sutra.mechanics.stress_v2 import derive_stress

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    args=ap.parse_args()
    project=Path(args.project_root).resolve()

    gatec=json.loads((project/"results"/"tranche3_2_gateC"/"gateC_certificate.json").read_text())
    if gatec.get("gateC_status")!="PASS": raise SystemExit("Gate C must PASS first.")

    outroot=project/"results"/"tranche3_4_regularization"
    outroot.mkdir(parents=True,exist_ok=True)
    reports=[]

    print("STRATA 0.3.5 | Tranche 3.4 | Mechanical regularization calibration")
    for k,rep in enumerate(gatec["sample_reports"],1):
        name=rep["sample"]; print(f"[{k}/{len(gatec['sample_reports'])}] {name}")
        src=project/"results"/"tranche3_2_gateC"/name
        edges=pd.read_parquet(src/"interface_mechanics.parquet")
        junctions=pd.read_parquet(src/"junction_geometry.parquet")
        pressure=pd.read_parquet(src/"cell_pressure.parquet")
        jr=pd.read_parquet(src/"junction_residuals.parquet")
        base_tau=edges.tension_like.to_numpy(float)
        base_resid=jr.fitted_residual.to_numpy(float)
        cell_ids=pressure.cell_id.astype(str).tolist()

        sweep,chosen,sol=calibrate_lambda(edges,cell_ids,len(junctions),base_tau,base_resid)
        out=outroot/name; out.mkdir(parents=True,exist_ok=True)
        sweep.to_csv(out/"lambda_sweep.csv",index=False)

        if chosen is None or sol is None:
            report={"sample":name,"status":"FAIL","reason":"no admissible regularization strength"}
            (out/"regularization_summary.json").write_text(json.dumps(report,indent=2))
            reports.append(report)
            print("    no admissible lambda -> FAIL")
            continue

        reg_edges=edges.copy(); reg_edges["tension_like"]=sol["tension"]
        reg_pressure=pressure.copy(); reg_pressure["pressure_like"]=sol["pressure"]

        # cells are only needed for stress
        sample_root=project/"data"/name
        hits=list(sample_root.rglob("*cells.parquet"))
        cells=pd.read_parquet(hits[0]); cells["cell_id"]=cells.cell_id.astype(str)
        stress=derive_stress(cells,reg_edges,reg_pressure)

        N=len(reg_edges); t=sol["tension"]
        neff=float(np.sum(t)**2/(np.sum(t*t)+1e-15))/max(N,1)
        k=max(1,int(np.ceil(0.01*N))); idx=np.argpartition(t,-k)[-k:]
        top1=float(np.sum(t[idx])/(np.sum(t)+1e-15))
        rho=float(sweep.loc[sweep.lambda_tau==chosen,"tension_spearman_vs_unregularized"].iloc[0])
        qratio=float(sweep.loc[sweep.lambda_tau==chosen,"q95_residual_ratio"].iloc[0])

        # We do not require a preconceived final distribution. The freeze
        # criterion asks for a material reduction in pathological concentration.
        baseline_neff=float((np.sum(base_tau)**2/(np.sum(base_tau*base_tau)+1e-15))/max(N,1))
        baseline_k=max(1,int(np.ceil(0.01*N))); bidx=np.argpartition(base_tau,-baseline_k)[-baseline_k:]
        baseline_top1=float(np.sum(base_tau[bidx])/(np.sum(base_tau)+1e-15))

        improved=(neff>=3*baseline_neff and top1<=0.8*baseline_top1)
        stable=(rho>=0.85 and qratio<=1.25)
        status="PASS" if improved and stable else "FAIL"

        report={
            "sample":name,"chosen_lambda_tau":chosen,
            "baseline_neff_fraction":baseline_neff,"regularized_neff_fraction":neff,
            "baseline_top1_mass":baseline_top1,"regularized_top1_mass":top1,
            "tension_spearman_vs_unregularized":rho,
            "q95_residual_ratio":qratio,
            "stress_valid_fraction":float(stress.stress_valid.mean()),
            "concentration_improved":bool(improved),"stability_preserved":bool(stable),
            "status":status,
        }
        reg_edges.to_parquet(out/"interface_mechanics_regularized.parquet",index=False)
        reg_pressure.to_parquet(out/"cell_pressure_regularized.parquet",index=False)
        stress.to_parquet(out/"cell_stress_regularized.parquet",index=False)
        (out/"regularization_summary.json").write_text(json.dumps(report,indent=2))
        reports.append(report)

        print(
            f"    lambda={chosen:.1e} Neff={100*baseline_neff:.2f}%->{100*neff:.2f}% "
            f"top1={100*baseline_top1:.1f}%->{100*top1:.1f}% "
            f"rho={rho:.3f} q95x={qratio:.3f} status={status}"
        )

    overall={
        "strata_version":"0.3.5","tranche":"3.4",
        "contract":"geometry-aware tension regularization calibration; no strain",
        "sample_reports":reports,
        "tranche3_4_status":"PASS" if all(r["status"]=="PASS" for r in reports) else "FAIL",
    }
    (outroot/"tranche3_4_certificate.json").write_text(json.dumps(overall,indent=2))
    print()
    print(f"Tranche 3.4: {overall['tranche3_4_status']}")
    print(f"Certificate: {outroot/'tranche3_4_certificate.json'}")

if __name__=="__main__":
    main()
