from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sutra.hierarchy.v071.resource_intake import sha256_file
from sutra.hierarchy.v072.pilot import (
    summarize_superedges,
    evaluate_candidates,
)
from sutra.hierarchy.v073.full import (
    FullConfig,
    maximal_disjoint_matching,
    contract_labels_inplace,
    failure_counts,
    hierarchy_node_sizes,
    size_summary,
    natural_stop_reason,
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")


def require(p:Path)->Path:
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def source_gate(project:Path):
    p=require(
        project/"results"/"hierarchy_v072_short_pilot"/
        "short_hierarchy_certificate.json"
    )
    d=json.loads(p.read_text())
    if (
        d.get("SHORT_HIERARCHY_GATE")!="PASS"
        or not d.get("LONGER_HIERARCHY_ALLOWED",False)
    ):
        raise RuntimeError("v0.7.2 does not allow the full hierarchy")
    return p,d


def _write_parquet(df,path):
    pq.write_table(
        pa.Table.from_pandas(df,preserve_index=False),
        path,
        compression="zstd",
    )


def _checkpoint(out:Path,level:int,labels:np.ndarray,trajectory:list):
    tmp=out/f".checkpoint_level{level}.npz.tmp"
    final=out/"checkpoint_latest.npz"
    with open(tmp,"wb") as f:
        np.savez_compressed(
            f,
            level=np.asarray([level],dtype=np.int64),
            labels=labels.astype(np.int64,copy=False),
        )
    os.replace(tmp,final)

    (out/"trajectory_partial.json").write_text(
        json.dumps(trajectory,indent=2)+"\n"
    )


def _load_resume(out:Path,n_cells:int):
    p=out/"checkpoint_latest.npz"
    if not p.exists():
        return 0,np.arange(n_cells,dtype=np.int64),[]
    z=np.load(p)
    level=int(z["level"][0])
    labels=np.asarray(z["labels"],dtype=np.int64)
    if len(labels)!=n_cells:
        raise RuntimeError("checkpoint cell count mismatch")
    t=out/"trajectory_partial.json"
    trajectory=json.loads(t.read_text()) if t.exists() else []
    return level,labels,trajectory


def one(project_s:str,sample:str,cfg_dict:dict,resume:bool):
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )

    project=Path(project_s)
    cfg=FullConfig(**cfg_dict)

    pilot=project/"results"/"hierarchy_v072_short_pilot"/sample
    edge_rel=pd.read_parquet(
        require(pilot/"level0_candidate_relations.parquet")
    )
    thresholds=json.loads(
        require(pilot/"frozen_thresholds.json").read_text()
    )
    level0_membership=pd.read_parquet(
        require(pilot/"membership_level0.parquet")
    )
    n_cells=int(len(level0_membership))

    out=project/"results"/"hierarchy_v073_full"/sample
    if out.exists() and not resume:
        shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)

    if resume:
        start_level,labels,trajectory=_load_resume(out,n_cells)
    else:
        start_level=0
        labels=np.arange(n_cells,dtype=np.int64)
        trajectory=[]

    merge_parts=[]
    if resume:
        for p in sorted(out.glob("merges_level*.parquet")):
            try:
                merge_parts.append(pd.read_parquet(p))
            except Exception:
                pass

    stop_reason=None
    hit_ceiling=False

    for level in range(start_level+1,cfg.max_levels+1):
        before_nodes=int(len(np.unique(labels)))
        S=summarize_superedges(edge_rel,labels)

        if len(S):
            S=S[S.n_boundary_edges>=cfg.min_boundary_edges].copy()
        C=evaluate_candidates(S,thresholds)
        SEL=maximal_disjoint_matching(C)

        n_adm=int(C.admissible.sum()) if len(C) else 0
        n_sel=int(SEL.selected.sum()) if len(SEL) else 0
        stop=natural_stop_reason(len(S),n_adm,n_sel)

        failures=failure_counts(C)
        ss_before=size_summary(labels)

        if stop is not None:
            entry={
                "level":level,
                "n_nodes_before":before_nodes,
                "n_superedges":int(len(S)),
                "n_admissible":n_adm,
                "n_selected":0,
                "n_nodes_after":before_nodes,
                "stop_reason":stop,
                **failures,
                **{f"before_{k}":v for k,v in ss_before.items()},
            }
            trajectory.append(entry)
            stop_reason=stop
            break

        selected=SEL[SEL.selected].copy()
        selected["level"]=level

        # Merge ledger carries the block values and margins of every accepted
        # production contraction.
        cols=[
            "level","super_i","super_j","n_boundary_edges",
            "molecular_z","mechanics_support_fraction",
            "abs_tension_z","abs_delta_p_z",
            "comm_strength","comm_reciprocity","ordering_merit",
        ]
        m=selected[cols].copy()
        m["survivor_node"]=np.minimum(
            m.super_i.astype(np.int64),m.super_j.astype(np.int64)
        )
        m["removed_node"]=np.maximum(
            m.super_i.astype(np.int64),m.super_j.astype(np.int64)
        )
        _write_parquet(m,out/f"merges_level{level:04d}.parquet")
        merge_parts.append(m)

        labels,_=contract_labels_inplace(labels,SEL)
        after_nodes=int(len(np.unique(labels)))
        ss_after=size_summary(labels)

        entry={
            "level":level,
            "n_nodes_before":before_nodes,
            "n_superedges":int(len(S)),
            "n_admissible":n_adm,
            "n_selected":n_sel,
            "n_nodes_after":after_nodes,
            "stop_reason":None,
            **failures,
            **{f"before_{k}":v for k,v in ss_before.items()},
            **{f"after_{k}":v for k,v in ss_after.items()},
        }
        trajectory.append(entry)

        if level%cfg.checkpoint_every==0:
            _checkpoint(out,level,labels,trajectory)

    else:
        hit_ceiling=True
        stop_reason="max_level_safety_ceiling"

    final_level=int(trajectory[-1]["level"]) if trajectory else 0

    # Always persist a final checkpoint/state.
    _checkpoint(out,final_level,labels,trajectory)

    final_membership=pd.DataFrame({
        "cell_index":np.arange(n_cells,dtype=np.int64),
        "hierarchy_node":labels.astype(np.int64),
        "final_level":final_level,
    })
    _write_parquet(final_membership,out/"final_membership.parquet")
    _write_parquet(
        hierarchy_node_sizes(labels),
        out/"final_node_sizes.parquet",
    )

    merges=(
        pd.concat(merge_parts,ignore_index=True)
        if merge_parts else pd.DataFrame()
    )
    _write_parquet(merges,out/"merge_trajectory.parquet")

    final_size=size_summary(labels)
    total_merges=int(len(merges))
    natural=bool(
        stop_reason in {
            "no_remaining_supernode_boundaries",
            "no_admissible_boundaries",
            "no_disjoint_admissible_contractions",
        }
    )

    report={
        "sample":sample,
        "n_level0_cells":n_cells,
        "n_level0_contact_edges":int(len(edge_rel)),
        "levels_executed":final_level,
        "productive_levels":int(sum(x["n_selected"]>0 for x in trajectory)),
        "total_contractions":total_merges,
        "final_nodes":int(final_size["n_nodes"]),
        "fraction_nodes_removed":float(
            (n_cells-final_size["n_nodes"])/n_cells
        ),
        "final_size_summary":final_size,
        "stop_reason":stop_reason,
        "natural_exhaustion":natural,
        "safety_ceiling_reached":hit_ceiling,
        "thresholds_frozen_from_level0":True,
        "scales_recomputed":False,
        "all_merges_from_admissible_matching":True,
        "trajectory":trajectory,
    }
    report["status"]="PASS" if natural and not hit_ceiling else "HOLD"

    (out/"full_summary.json").write_text(
        json.dumps(report,indent=2)+"\n"
    )
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--max-levels",type=int,default=250)
    ap.add_argument("--checkpoint-every",type=int,default=5)
    ap.add_argument("--resume",action="store_true")
    a=ap.parse_args()

    if a.max_levels<1:
        raise SystemExit("--max-levels must be positive")
    if a.checkpoint_every<1:
        raise SystemExit("--checkpoint-every must be positive")

    project=Path(a.project_root).resolve()
    source_path,_=source_gate(project)

    cfg=FullConfig(
        max_levels=a.max_levels,
        checkpoint_every=a.checkpoint_every,
    )

    print("STRATA 0.7.3 | Full hierarchy")
    print("v0.7.2 scientific gates frozen; pilot-only contraction cap removed.")
    print("Each level uses deterministic maximal disjoint admissible matching.")
    print("Natural exhaustion required for PASS.")
    print(
        f"max_levels(safety)={cfg.max_levels} "
        f"checkpoint_every={cfg.checkpoint_every} "
        f"resume={a.resume}"
    )
    print(f"Parallel specimens: {min(a.workers,len(SAMPLES))}\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(SAMPLES))) as ex:
        futs={
            ex.submit(
                one,str(project),s,cfg.__dict__.copy(),a.resume
            ):s
            for s in SAMPLES
        }
        for f in as_completed(futs):
            r=f.result(); reports.append(r)
            print(
                f"[DONE] {r['sample']}: "
                f"levels={r['levels_executed']} "
                f"merges={r['total_contractions']:,} "
                f"nodes={r['n_level0_cells']:,}->{r['final_nodes']:,} "
                f"removed={100*r['fraction_nodes_removed']:.2f}% "
                f"stop={r['stop_reason']} {r['status']}",
                flush=True,
            )

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)

    out=project/"results"/"hierarchy_v073_full"
    out.mkdir(parents=True,exist_ok=True)
    cert={
        "strata_version":"0.7.3",
        "stage":"full hierarchy",
        "source_short_hierarchy_certificate_sha256":sha256_file(source_path),
        "source_short_hierarchy_gate":"PASS",
        "production_policy":{
            "scientific_admissibility":"unchanged from v0.7.2",
            "matching":"deterministic maximal disjoint matching",
            "pilot_pair_cap_removed":True,
            "thresholds":"frozen Level-0 thresholds",
            "later_level_rescaling":False,
            "missing_mechanics":"missing, never zero",
            "measured_contact_support_required":True,
            "pass_condition":"natural exhaustion only",
            "max_levels":cfg.max_levels,
            "checkpoint_every":cfg.checkpoint_every,
        },
        "sample_reports":reports,
        "FULL_HIERARCHY_GATE":"PASS" if gate else "HOLD",
        "HIERARCHY_TRAJECTORY_COMPLETE":bool(gate),
    }
    p=out/"full_hierarchy_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print(f"\nFULL HIERARCHY GATE: {cert['FULL_HIERARCHY_GATE']}")
    print(
        "HIERARCHY TRAJECTORY COMPLETE: "
        f"{cert['HIERARCHY_TRAJECTORY_COMPLETE']}"
    )
    print(f"Certificate: {p}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
