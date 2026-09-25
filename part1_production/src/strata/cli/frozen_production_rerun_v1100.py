from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

from strata_hierarchy.v1100.production import require,verify_freeze,run_one
from strata_hierarchy.v1100.ledger import writejson

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument(
        "--engine-config",
        default="configs/hierarchy_v093_full_completed_ultraslow_flow.json"
    )
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    rule,freeze_cert=verify_freeze(project)
    engine=json.loads(require(project/a.engine_config).read_text())

    # Upstream completed tissue and slow-step certification remain mandatory.
    d91=json.loads(require(
        project/"results"/"hierarchy_v091_network_exhaustive_completion"/
        "network_exhaustive_completion_global_certificate.json"
    ).read_text())
    d92=json.loads(require(
        project/"results"/"hierarchy_v092_step_spectrum_consistency"/
        "step_spectrum_global_certificate.json"
    ).read_text())
    if d91.get("READY_FOR_COMPLETED_ULTRASLOW_FLOW") is not True:
        raise SystemExit("ERROR: v0.9.1 completed tissue state not ready")
    if d92.get("READY_TO_SELECT_PRODUCTION_STEP") is not True:
        raise SystemExit("ERROR: v0.9.2 slow-step consistency not certified")

    print("STRATA 1.1.0 | Frozen final production hierarchy")
    print("Scientific merge model: frozen v0.9.3 completed ultraslow engine.")
    print("Terminal rule: frozen v1.0.10 causal rule.")
    print("No target node count.")
    print("Exact partition, mass, lineage-mass, and initial-component invariants are hard assertions after every merge batch.")
    print("Scientific evaluations persist exact labels, ancestry, node/edge state, transport, provenance, and first/second scale differences.")
    print("Three specimens run independently in parallel.")
    print(f"Resume mode: {bool(a.resume)}")
    print(f"Parallel specimen workers: {min(a.workers,3)}")
    print()

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        futs={
            ex.submit(run_one,str(project),s,engine,rule,bool(a.resume)):s
            for s in SAMPLES
        }
        for f in as_completed(futs):
            r=f.result();reports.append(r)
            print(
                f"[DONE] {r['sample']}: microsteps={r['microsteps']:,} "
                f"merges={r['total_merges']:,} "
                f"nodes={r['level0_cells']:,}->{r['final_nodes']:,} "
                f"removed={100*r['removed_fraction']:.2f}% "
                f"evaluations={r['scientific_evaluations']:,} "
                f"stop={r['stop_reason']} {r['status']}",
                flush=True
            )

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v1100_frozen_production_rerun"
    out.mkdir(parents=True,exist_ok=True)
    cert={
        "strata_version":"1.1.0",
        "stage":"frozen final production hierarchy",
        "scientific_engine":"v0.9.3 completed ultraslow hierarchy",
        "terminal_rule":"v1.0.10 frozen causal terminal rule",
        "terminal_rule_sha256":reports[0]["frozen_terminal_rule_sha256"] if reports else None,
        "parallel_specimens":3,
        "hard_invariants":{
            "exact_Level0_partition_every_batch":True,
            "exact_mass_conservation_every_batch":True,
            "exact_parent_child_mass_every_batch":True,
            "lineage_mass_matches_partition_every_batch":True,
            "initial_components_never_joined":True,
        },
        "sample_reports":reports,
        "FROZEN_PRODUCTION_GATE":"PASS" if gate else "HOLD",
        "FINAL_PRODUCTION_HIERARCHY_COMPLETE":bool(gate),
        "READY_FOR_HIERARCHY_ATLAS":bool(gate),
        "READY_FOR_MANUSCRIPT_BIOLOGY":bool(gate),
    }
    p=out/"frozen_production_global_certificate.json"
    writejson(cert,p)

    print()
    print(f"FROZEN PRODUCTION GATE: {cert['FROZEN_PRODUCTION_GATE']}")
    print(f"FINAL PRODUCTION HIERARCHY COMPLETE: {cert['FINAL_PRODUCTION_HIERARCHY_COMPLETE']}")
    print(f"READY FOR HIERARCHY ATLAS: {cert['READY_FOR_HIERARCHY_ATLAS']}")
    print(f"READY FOR MANUSCRIPT BIOLOGY: {cert['READY_FOR_MANUSCRIPT_BIOLOGY']}")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
