from __future__ import annotations
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sutra.hierarchy.v092.sweep import (
    run_schedule,partition_pair_agreement,label_overlap_jaccard,
    interpolate_history
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def require(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(p)
    return p

def writepq(df,path):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),path,compression="zstd")

def one(project_s,sample,cfg):
    os.environ.update(
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1"
    )
    project=Path(project_s)
    src=project/"results"/"hierarchy_v091_network_exhaustive_completion"/sample
    edges=pd.read_parquet(require(src/"completed_tissue_backbone_edges.parquet"))
    n_cells=int(json.loads(require(src/"network_exhaustive_completion_certificate.json").read_text())["n_cells"])

    fracs=[float(x) for x in cfg["fractions_fast_to_slow"]]
    weights=cfg["weights"]
    target=float(cfg["pilot_target_removed_fraction"])

    out=project/"results"/"hierarchy_v092_step_spectrum_consistency"/sample
    out.mkdir(parents=True,exist_ok=True)

    labels_by={}
    histories={}
    summaries=[]
    for f in fracs:
        labels,hist,s=run_schedule(
            edges,n_cells,f,weights,
            max_steps=int(cfg["max_steps"]),
            target_removed=target,
        )
        key=f"{f:.6g}"
        labels_by[key]=labels
        histories[key]=hist
        summaries.append(s)
        np.savez_compressed(out/f"labels_fraction_{key}.npz",labels=labels)
        writepq(hist,out/f"history_fraction_{key}.parquet")

    slow_key=f"{fracs[-1]:.6g}"
    slow_labels=labels_by[slow_key]
    comp=[]
    for f in fracs:
        key=f"{f:.6g}"
        comp.append({
            "fraction":f,
            "pair_agreement_vs_slowest":partition_pair_agreement(labels_by[key],slow_labels),
            "mean_block_jaccard_vs_slowest":label_overlap_jaccard(labels_by[key],slow_labels),
        })
    compdf=pd.DataFrame(comp)
    writepq(compdf,out/"partition_consistency.parquet")

    grid=np.linspace(0,target,41)
    rows=[]
    slow_interp=interpolate_history(histories[slow_key],grid)
    for f in fracs:
        key=f"{f:.6g}"
        h=interpolate_history(histories[key],grid)
        for c in ["pair_cost_mean","pair_cost_q95","candidate_superedges","nodes_after"]:
            denom=np.maximum(np.abs(slow_interp[c].to_numpy(float)),1e-12)
            rel=np.abs(h[c].to_numpy(float)-slow_interp[c].to_numpy(float))/denom
            rows.append({
                "fraction":f,
                "observable":c,
                "mean_relative_drift_vs_slowest":float(np.mean(rel)),
                "max_relative_drift_vs_slowest":float(np.max(rel)),
            })
    drift=pd.DataFrame(rows)
    writepq(drift,out/"trajectory_consistency.parquet")

    sumdf=pd.DataFrame(summaries)
    writepq(sumdf,out/"schedule_summary.parquet")

    # Convergence criterion focuses on the three slowest schedules.
    slow3=fracs[-3:]
    slow3_keys=[f"{x:.6g}" for x in slow3]
    slow3_agree=[
        partition_pair_agreement(labels_by[k],slow_labels)
        for k in slow3_keys
    ]
    slow3_j=[
        label_overlap_jaccard(labels_by[k],slow_labels)
        for k in slow3_keys
    ]
    converged=bool(
        min(slow3_agree)>=float(cfg["min_pair_agreement_slow_regime"])
        and min(slow3_j)>=float(cfg["min_block_jaccard_slow_regime"])
    )

    report={
        "sample":sample,
        "fractions_fast_to_slow":fracs,
        "pilot_target_removed_fraction":target,
        "schedule_summaries":summaries,
        "slow_regime_fractions":slow3,
        "slow_regime_pair_agreement_min":float(min(slow3_agree)),
        "slow_regime_block_jaccard_min":float(min(slow3_j)),
        "slow_regime_converged":converged,
        "status":"PASS" if converged else "HOLD",
    }
    (out/"step_spectrum_certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v092_step_spectrum.json")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    cfg=json.loads(require(project/a.config).read_text())

    cert=require(project/"results"/"hierarchy_v091_network_exhaustive_completion"/"network_exhaustive_completion_global_certificate.json")
    d=json.loads(cert.read_text())
    if d.get("READY_FOR_COMPLETED_ULTRASLOW_FLOW") is not True:
        raise SystemExit("ERROR: v0.9.1 completed tissue state is not ready")

    print("STRATA 0.9.2 | Step-spectrum consistency sweep")
    print("Scientific model frozen; only contraction fraction varies.")
    print("Fast -> slow schedules are run sequentially within each specimen.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        futs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(futs):
            r=f.result(); reports.append(r)
            print(
                f"[DONE] {r['sample']}: "
                f"slow_agreement={r['slow_regime_pair_agreement_min']:.5f} "
                f"slow_J={r['slow_regime_block_jaccard_min']:.5f} "
                f"{r['status']}",
                flush=True
            )

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v092_step_spectrum_consistency"
    g={
        "strata_version":"0.9.2",
        "stage":"completed-field contraction step-spectrum consistency",
        "sample_reports":reports,
        "STEP_SPECTRUM_GATE":"PASS" if gate else "HOLD",
        "SLOW_LIMIT_CONSISTENT":bool(gate),
        "READY_TO_SELECT_PRODUCTION_STEP":bool(gate),
    }
    p=out/"step_spectrum_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nSTEP-SPECTRUM GATE: {g['STEP_SPECTRUM_GATE']}")
    print(f"SLOW LIMIT CONSISTENT: {g['SLOW_LIMIT_CONSISTENT']}")
    print(f"READY TO SELECT PRODUCTION STEP: {g['READY_TO_SELECT_PRODUCTION_STEP']}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
