from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from strata.preflight.core import PreflightConfig, synthetic_recovery_suite
from strata.preflight.sample_audit import audit_sample
from strata.preflight.cross_sample import cross_sample_audit


def _strip_private(r):
    return {k:v for k,v in r.items() if not k.startswith("_")}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--sample-workers",type=int,default=None)
    args=ap.parse_args()

    project=Path(args.project_root).resolve()
    cfg_path=project/"configs"/"preflight.json"
    cfgj=json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    if args.sample_workers is not None:
        cfgj["sample_workers"]=args.sample_workers
    cfg=PreflightConfig(**cfgj)

    # Resolve samples from Gate B if present, otherwise data subdirectories.
    gb=project/"results"/"tranche2_1_gateB"/"gateB_certificate.json"
    if gb.exists():
        x=json.loads(gb.read_text())
        samples=[r["sample"] for r in x.get("sample_certificates",[])]
    else:
        samples=sorted(p.name for p in (project/"data").iterdir() if p.is_dir())

    if not samples:
        raise SystemExit("No STRATA samples discovered.")

    outroot=project/"results"/"preflight"
    outroot.mkdir(parents=True,exist_ok=True)

    print("STRATA Preflight 0.1.0 | Level-0 scientific integrity survey")
    print(f"Project: {project}")
    print(f"Samples: {len(samples)}")
    print(f"Parallel specimen workers: {min(cfg.sample_workers,len(samples))}")
    print()

    t0=time.time()
    reports=[]
    workers=min(max(1,cfg.sample_workers),len(samples))
    cfg_dict=cfg.__dict__.copy()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(audit_sample,str(project),s,cfg_dict):s for s in samples}
        for fut in as_completed(futs):
            s=futs[fut]
            try:
                r=fut.result()
            except Exception as e:
                r={
                    "sample":s,
                    "data_integrity_status":"FAIL",
                    "structural_status":"FAIL",
                    "numerical_robustness_status":"FAIL",
                    "fatal_error":f"{type(e).__name__}: {e}"
                }
            reports.append(r)
            print(
                f"[DONE] {s}: "
                f"data={r.get('data_integrity_status','FAIL')} "
                f"struct={r.get('structural_status','FAIL')} "
                f"numeric={r.get('numerical_robustness_status','FAIL')} "
                f"{r.get('runtime_seconds',0):.1f}s",
                flush=True
            )

    order={s:i for i,s in enumerate(samples)}
    reports.sort(key=lambda r:order.get(r["sample"],999))
    cross=cross_sample_audit(reports)
    synth=synthetic_recovery_suite(cfg.synthetic_seed)

    # Scientific adequacy: conservative. Any large unassigned geometry or missing
    # morphology/transcripts becomes WARN, not automatic FAIL.
    sci_warn=False
    sci_reasons=[]
    for r in reports:
        gm=r.get("geometry_coverage",{})
        if gm.get("available") and gm.get("internal_unassigned_fraction",0)>cfg.max_internal_unassigned_fraction_warn:
            sci_warn=True
            sci_reasons.append(
                f"{r['sample']}: internal unassigned geometry "
                f"{100*gm['internal_unassigned_fraction']:.1f}%"
            )
        if not r.get("morphology_files"):
            sci_warn=True
            sci_reasons.append(f"{r['sample']}: morphology unavailable")
        if not r.get("transcripts",{}).get("available",False):
            sci_warn=True
            sci_reasons.append(f"{r['sample']}: transcript table unavailable")

    if synth["recovery_fraction"]<cfg.min_synthetic_recovery_warn:
        sci_warn=True
        sci_reasons.append("synthetic recovery suite below threshold")

    data_fail=any(r.get("data_integrity_status")=="FAIL" for r in reports) or cross["gene_panel"]["status"]=="FAIL"
    structural_fail=any(r.get("structural_status")=="FAIL" for r in reports)
    numerical_fail=any(r.get("numerical_robustness_status")=="FAIL" for r in reports)

    category_status={
        "data_integrity":"FAIL" if data_fail else "PASS",
        "mathematical_structural_validity":"FAIL" if structural_fail else (
            "WARN" if any(r.get("structural_status")=="WARN" for r in reports) else "PASS"
        ),
        "numerical_robustness":"FAIL" if numerical_fail else (
            "WARN" if any(r.get("numerical_robustness_status")=="WARN" for r in reports) else "PASS"
        ),
        "scientific_identifiability_adequacy":"WARN" if sci_warn else "PASS",
        "synthetic_recovery":"PASS" if synth["recovery_fraction"]>=cfg.min_synthetic_recovery_warn else "WARN",
    }

    hard_fail=any(v=="FAIL" for v in category_status.values())
    unresolved_warn=[k for k,v in category_status.items() if v=="WARN"]

    # A clean freeze requires zero hard failures and zero unresolved warnings.
    freeze_status="PASS" if (not hard_fail and not unresolved_warn) else (
        "FAIL" if hard_fail else "HOLD"
    )

    overall={
        "strata_preflight_version":"0.1.0",
        "project_root":str(project),
        "samples":samples,
        "config":cfg.__dict__,
        "category_status":category_status,
        "level0_freeze_status":freeze_status,
        "freeze_rule":"PASS requires no FAIL and no unresolved WARN categories.",
        "scientific_warning_reasons":sci_reasons,
        "cross_sample":cross,
        "synthetic_recovery":synth,
        "sample_reports":[_strip_private(r) for r in reports],
        "runtime_seconds":float(time.time()-t0),
    }

    (outroot/"preflight_certificate.json").write_text(json.dumps(overall,indent=2))
    for r in reports:
        (outroot/f"{r['sample']}_preflight.json").write_text(
            json.dumps(_strip_private(r),indent=2)
        )

    print()
    print("Category status")
    for k,v in category_status.items():
        print(f"  {k}: {v}")
    print()
    print(f"LEVEL-0 FREEZE: {freeze_status}")
    print(f"Certificate: {outroot/'preflight_certificate.json'}")


if __name__=="__main__":
    main()
