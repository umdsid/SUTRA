
from __future__ import annotations
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from sutra.hierarchy.v120.atlas import write_preflight

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default=".")
    ap.add_argument("--config",default="configs/hierarchy_v120_merger_driver_atlas.json")
    a=ap.parse_args()
    project=Path(a.project).resolve()
    cfg=json.loads((project/a.config).read_text())
    outdir=project/"results"/"hierarchy_v120_merger_driver_atlas"
    outdir.mkdir(parents=True,exist_ok=True)

    print("STRATA 1.2.0 | Hierarchy merger-driver atlas")
    print("Final v1.1.2 hierarchy is frozen and unchanged.")
    print("For each final landmark, resolving exact Level-0 entry/center partitions.")
    print("Actual component mergers are reconstructed from nested ancestry, not inferred from proximity.")
    print("Native Xenium expression/coordinate sources are inventoried for gene-driver analysis.")
    print("No new merges, hierarchy changes, or scientific thresholds.")
    print(f"Parallel specimen workers: {cfg['parallel_specimens']}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=cfg["parallel_specimens"]) as ex:
        fs={ex.submit(write_preflight,project,s,cfg,outdir):s for s in cfg["samples"]}
        for f in as_completed(fs):
            r=f.result(); reps.append(r)
            print(
                f"[DONE] {r['sample']}: N0={r.get('n0')} "
                f"landmarks={r.get('landmarks')} "
                f"resolved_transitions={r.get('resolved_transitions', 0)} "
                f"replay_stream={r.get('replay_stream_ready', False)} "
                f"status={r.get('status', 'HOLD')}"
            )

    reps=sorted(reps,key=lambda x:x["sample"])
    structure=all(r.get("structure_ready",False) for r in reps)
    expr=all(r.get("expression_ready",False) for r in reps)
    gate="PASS" if structure else "HOLD"
    cert={
      "stage":"hierarchy merger-driver atlas preflight",
      "strata_version":"1.2.0",
      "MERGER_DRIVER_PREFLIGHT_GATE":gate,
      "EXACT_LANDMARK_PARTITIONS_READY":structure,
      "UNIQUE_NATIVE_EXPRESSION_SOURCES_READY":expr,
      "HIERARCHY_CHANGED":False,
      "NEW_MERGES_PERFORMED":False,
      "THRESHOLDS_MODIFIED":False,
      "READY_FOR_GENE_DRIVER_SCORING":structure and expr,
      "sample_reports":reps
    }
    cp=outdir/"merger_driver_atlas_global_certificate.json"
    cp.write_text(json.dumps(cert,indent=2))
    print(f"\nMERGER DRIVER PREFLIGHT GATE: {gate}")
    print(f"EXACT LANDMARK PARTITIONS READY: {structure}")
    print(f"UNIQUE NATIVE EXPRESSION SOURCES READY: {expr}")
    print(f"READY FOR GENE DRIVER SCORING: {structure and expr}")
    print(f"Certificate: {cp}")

if __name__=="__main__":
    main()
