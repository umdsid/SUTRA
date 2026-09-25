from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sutra.hierarchy.v071.block_preflight import (
    read_10x_cells_by_genes,
    normalized_expression_dense,
)
from sutra.hierarchy.v071.resource_intake import sha256_file
from sutra.hierarchy.v072.pilot import (
    PilotConfig,
    communication_relations,
    freeze_thresholds,
    summarize_superedges,
    evaluate_candidates,
    greedy_matching,
    contract_labels,
    membership_table,
    level_summary,
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")


def require(p:Path)->Path:
    if not p.exists(): raise FileNotFoundError(p)
    return p


def audit_active_gate(project:Path):
    p=require(
        project/"results"/"hierarchy_v071_active_blocks"/
        "active_block_certificate.json"
    )
    d=json.loads(p.read_text())
    if d.get("ACTIVE_BLOCK_GATE")!="PASS" or not d.get("SHORT_HIERARCHY_READY",False):
        raise RuntimeError("v0.7.1 active block gate is not short-hierarchy ready")
    return p,d


def one(project_s:str,sample:str,cfg_dict:dict):
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    project=Path(project_s)
    cfg=PilotConfig(**cfg_dict)

    l0=project/"results"/"hierarchy_level0_v070"/sample
    v71=project/"results"/"hierarchy_v071_active_blocks"/sample
    E=pd.read_parquet(require(v71/"contact_block_relations.parquet"))
    registry=pd.read_parquet(require(v71/"canonical_signaling_registry.parquet"))
    cells=pd.read_parquet(require(l0/"cells.parquet"))

    manifest=json.loads(require(l0/"expression_manifest.json").read_text())
    matrix_path=Path(manifest["source_matrix"])
    if sha256_file(matrix_path)!=manifest["source_matrix_sha256"]:
        raise RuntimeError(f"{sample}: expression source hash changed")

    X,barcodes,genes=read_10x_cells_by_genes(matrix_path)
    if len(barcodes)!=len(cells):
        raise RuntimeError(f"{sample}: expression/cell count mismatch")
    Y=normalized_expression_dense(X)

    C=communication_relations(Y,E,tuple(map(str,genes)),registry,cfg.eps)
    edge_rel=pd.concat(
        [E.reset_index(drop=True),C.reset_index(drop=True)],axis=1
    )

    thresholds=freeze_thresholds(edge_rel,cfg)

    out=project/"results"/"hierarchy_v072_short_pilot"/sample
    out.mkdir(parents=True,exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(edge_rel,preserve_index=False),
        out/"level0_candidate_relations.parquet",
        compression="zstd",
    )
    (out/"frozen_thresholds.json").write_text(
        json.dumps(thresholds,indent=2)+"\n"
    )

    labels=np.arange(len(cells),dtype=np.int64)
    ledgers=[];summaries=[]

    # Level 0 membership written for completeness.
    pq.write_table(
        pa.Table.from_pandas(membership_table(labels,0),preserve_index=False),
        out/"membership_level0.parquet",
        compression="zstd",
    )

    for level in range(1,cfg.n_levels+1):
        before=labels.copy()
        S=summarize_superedges(edge_rel,labels)
        CAND=evaluate_candidates(S,thresholds)

        active_nodes=len(np.unique(labels))
        SEL=greedy_matching(CAND,active_nodes,cfg)

        selected_keys=set()
        if len(SEL):
            selected_keys=set(
                zip(
                    SEL.loc[SEL.selected,"super_i"].astype(int),
                    SEL.loc[SEL.selected,"super_j"].astype(int),
                )
            )

        if len(CAND):
            CAND=CAND.copy()
            CAND["selected"]=[
                (int(a),int(b)) in selected_keys
                for a,b in zip(CAND.super_i,CAND.super_j)
            ]
            CAND["level"]=level
            ledgers.append(CAND)

        labels,_=contract_labels(labels,SEL)
        summaries.append(
            level_summary(level,CAND,SEL,before,labels)
        )

        pq.write_table(
            pa.Table.from_pandas(
                membership_table(labels,level),preserve_index=False
            ),
            out/f"membership_level{level}.parquet",
            compression="zstd",
        )

    ledger=(
        pd.concat(ledgers,ignore_index=True)
        if ledgers else pd.DataFrame()
    )
    pq.write_table(
        pa.Table.from_pandas(ledger,preserve_index=False),
        out/"candidate_ledger.parquet",
        compression="zstd",
    )

    total_selected=sum(x["n_selected"] for x in summaries)
    final_nodes=int(len(np.unique(labels)))
    all_nonexpanding=all(
        s["n_nodes_after"]<=s["n_nodes_before"] for s in summaries
    )
    no_empty=all(s["n_nodes_after"]>0 for s in summaries)

    report={
        "sample":sample,
        "n_level0_cells":int(len(cells)),
        "n_level0_contact_edges":int(len(E)),
        "n_panel_supported_interactions":int(
            registry.panel_supported.astype(bool).sum()
            if "panel_supported" in registry else len(registry)
        ),
        "levels":summaries,
        "total_selected_contractions":int(total_selected),
        "final_nodes":final_nodes,
        "fraction_nodes_removed":float((len(cells)-final_nodes)/len(cells)),
        "thresholds_frozen_from_level0":True,
        "scales_recomputed":False,
        "all_contractions_conjunctively_admissible":bool(
            len(ledger)==0 or
            ledger.loc[ledger.selected,"admissible"].all()
        ),
        "monotone_node_count":bool(all_nonexpanding),
        "nonempty_hierarchy":bool(no_empty),
    }
    report["status"]="PASS" if (
        report["all_contractions_conjunctively_admissible"]
        and report["monotone_node_count"]
        and report["nonempty_hierarchy"]
        and total_selected>0
    ) else "HOLD"

    (out/"pilot_summary.json").write_text(
        json.dumps(report,indent=2)+"\n"
    )
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--levels",type=int,default=3)
    ap.add_argument("--max-pair-fraction",type=float,default=0.005)
    ap.add_argument("--max-pairs-per-level",type=int,default=250)
    ap.add_argument("--min-mechanics-support-fraction",type=float,default=0.50)
    a=ap.parse_args()

    if a.levels<1 or a.levels>5:
        raise SystemExit("--levels must be between 1 and 5 for the short pilot")
    if not (0<a.max_pair_fraction<=0.02):
        raise SystemExit("--max-pair-fraction must be in (0,0.02]")
    if not (0<=a.min_mechanics_support_fraction<=1):
        raise SystemExit("--min-mechanics-support-fraction must be in [0,1]")

    project=Path(a.project_root).resolve()
    active_path,_=audit_active_gate(project)

    cfg=PilotConfig(
        n_levels=a.levels,
        max_pair_fraction=a.max_pair_fraction,
        max_pairs_per_level=a.max_pairs_per_level,
        min_mechanics_support_fraction=a.min_mechanics_support_fraction,
    )
    cfg_dict=cfg.__dict__.copy()

    print("STRATA 0.7.2 | Short hierarchy pilot")
    print("Three active blocks are conjunctive; no compensatory weighted gate.")
    print("All thresholds are frozen from Level-0.")
    print(
        f"levels={cfg.n_levels} max_pair_fraction={cfg.max_pair_fraction} "
        f"max_pairs/level={cfg.max_pairs_per_level}"
    )
    print(f"Parallel specimens: {min(a.workers,len(SAMPLES))}\n")

    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(SAMPLES))) as ex:
        futs={
            ex.submit(one,str(project),s,cfg_dict):s
            for s in SAMPLES
        }
        for f in as_completed(futs):
            r=f.result();reports.append(r)
            print(
                f"[DONE] {r['sample']}: "
                f"contractions={r['total_selected_contractions']} "
                f"removed={100*r['fraction_nodes_removed']:.2f}% "
                f"final_nodes={r['final_nodes']:,} "
                f"{r['status']}",
                flush=True,
            )
            for x in r["levels"]:
                print(
                    f"    L{x['level']}: nodes "
                    f"{x['n_nodes_before']:,}->{x['n_nodes_after']:,} "
                    f"admissible={x['n_admissible']:,} "
                    f"selected={x['n_selected']:,}",
                    flush=True,
                )

    reports.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reports)
    out=project/"results"/"hierarchy_v072_short_pilot"
    out.mkdir(parents=True,exist_ok=True)
    cert={
        "strata_version":"0.7.2",
        "stage":"short hierarchy pilot",
        "source_active_block_certificate_sha256":sha256_file(active_path),
        "source_active_block_gate":"PASS",
        "policy":{
            "eligibility":"conjunctive molecular + mechanics + directed communication",
            "measured_contact_support_required":True,
            "missing_mechanics":"missing, never zero",
            "thresholds":"frozen from Level-0 only",
            "later_level_rescaling":False,
            "pilot_levels":cfg.n_levels,
            "max_pair_fraction":cfg.max_pair_fraction,
            "max_pairs_per_level":cfg.max_pairs_per_level,
            "min_mechanics_support_fraction":cfg.min_mechanics_support_fraction,
        },
        "sample_reports":reports,
        "SHORT_HIERARCHY_GATE":"PASS" if gate else "HOLD",
        "LONGER_HIERARCHY_ALLOWED":bool(gate),
    }
    p=out/"short_hierarchy_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print(f"\nSHORT HIERARCHY GATE: {cert['SHORT_HIERARCHY_GATE']}")
    print(f"LONGER HIERARCHY ALLOWED: {cert['LONGER_HIERARCHY_ALLOWED']}")
    print(f"Certificate: {p}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
