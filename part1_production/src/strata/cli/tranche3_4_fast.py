from __future__ import annotations
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd

from strata.mechanics.warmstart_regularization import adaptive_warm_calibration
from strata.mechanics.stress_v2 import derive_stress


def _run_sample(payload):
    project_s,rep=payload
    project=Path(project_s);name=rep["sample"];t0=time.time()
    src=project/"results"/"tranche3_2_gateC"/name
    edges=pd.read_parquet(src/"interface_mechanics.parquet")
    junctions=pd.read_parquet(src/"junction_geometry.parquet")
    pressure=pd.read_parquet(src/"cell_pressure.parquet")
    jr=pd.read_parquet(src/"junction_residuals.parquet")

    base_tau=edges.tension_like.to_numpy(float)
    base_p=pressure.pressure_like.to_numpy(float)
    base_resid=jr.fitted_residual.to_numpy(float)
    ids=pressure.cell_id.astype(str).tolist()

    sweep,chosen,sol,cache=adaptive_warm_calibration(
        edges,ids,len(junctions),base_tau,base_p,base_resid
    )
    out=project/"results"/"tranche3_4_regularization_fast"/name
    out.mkdir(parents=True,exist_ok=True)
    sweep.to_csv(out/"lambda_sweep.csv",index=False)

    if chosen is None:
        r={"sample":name,"status":"FAIL","reason":"no admissible lambda","runtime_seconds":time.time()-t0}
        (out/"regularization_summary.json").write_text(json.dumps(r,indent=2))
        return r

    reg_edges=edges.copy();reg_edges["tension_like"]=sol["tension"]
    reg_pressure=pressure.copy();reg_pressure["pressure_like"]=sol["pressure"]

    cells=pd.read_parquet(list((project/"data"/name).rglob("*cells.parquet"))[0])
    cells["cell_id"]=cells.cell_id.astype(str)
    stress=derive_stress(cells,reg_edges,reg_pressure)

    N=len(edges);t=sol["tension"]
    neff=float(np.sum(t)**2/(np.sum(t*t)+1e-15))/N
    k=max(1,int(np.ceil(.01*N)));idx=np.argpartition(t,-k)[-k:]
    top1=float(np.sum(t[idx])/(np.sum(t)+1e-15))
    bneff=float(np.sum(base_tau)**2/(np.sum(base_tau*base_tau)+1e-15))/N
    bidx=np.argpartition(base_tau,-k)[-k:]
    btop1=float(np.sum(base_tau[bidx])/(np.sum(base_tau)+1e-15))
    row=sweep.loc[sweep.lambda_tau==chosen].iloc[0]
    rho=float(row.tension_spearman_vs_unregularized);qratio=float(row.q95_residual_ratio)

    improved=(neff>=3*bneff and top1<=.8*btop1)
    stable=(rho>=.85 and qratio<=1.25)
    status="PASS" if improved and stable else "FAIL"
    r={
        "sample":name,"chosen_lambda_tau":chosen,
        "baseline_neff_fraction":bneff,"regularized_neff_fraction":neff,
        "baseline_top1_mass":btop1,"regularized_top1_mass":top1,
        "tension_spearman_vs_unregularized":rho,"q95_residual_ratio":qratio,
        "stress_valid_fraction":float(stress.stress_valid.mean()),
        "concentration_improved":bool(improved),"stability_preserved":bool(stable),
        "n_lambda_solves":len(sweep),"runtime_seconds":time.time()-t0,
        "status":status
    }
    reg_edges.to_parquet(out/"interface_mechanics_regularized.parquet",index=False)
    reg_pressure.to_parquet(out/"cell_pressure_regularized.parquet",index=False)
    stress.to_parquet(out/"cell_stress_regularized.parquet",index=False)
    (out/"regularization_summary.json").write_text(json.dumps(r,indent=2))
    return r


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--sample-workers",type=int,default=3)
    args=ap.parse_args()
    project=Path(args.project_root).resolve()
    gc=json.loads((project/"results"/"tranche3_2_gateC"/"gateC_certificate.json").read_text())
    if gc.get("gateC_status")!="PASS":raise SystemExit("Gate C must PASS first.")
    outroot=project/"results"/"tranche3_4_regularization_fast";outroot.mkdir(parents=True,exist_ok=True)

    reps=gc["sample_reports"];workers=min(max(args.sample_workers,1),len(reps))
    print("STRATA 0.4.0 | Tranche 3.4 | Warm-started continuation + parallel specimens")
    print(f"Running {len(reps)} specimens on {workers} workers.",flush=True)
    reports=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(_run_sample,(str(project),r)):r["sample"] for r in reps}
        for fut in as_completed(futs):
            name=futs[fut]
            try:r=fut.result()
            except Exception as e:r={"sample":name,"status":"FAIL","reason":f"{type(e).__name__}: {e}"}
            reports.append(r)
            print(f"[DONE] {name}: {r.get('status')} {r.get('reason','')}",flush=True)

    order={r["sample"]:i for i,r in enumerate(reps)}
    reports.sort(key=lambda r:order.get(r["sample"],999))
    overall={
        "strata_version":"0.4.0","tranche":"3.4",
        "contract":"warm-started geometry-aware tension regularization; no strain",
        "sample_reports":reports,
        "tranche3_4_status":"PASS" if all(r.get("status")=="PASS" for r in reports) else "FAIL"
    }
    (outroot/"tranche3_4_certificate.json").write_text(json.dumps(overall,indent=2))
    print()
    print(f"Tranche 3.4: {overall['tranche3_4_status']}")
    print(f"Certificate: {outroot/'tranche3_4_certificate.json'}")

if __name__=="__main__":
    main()
