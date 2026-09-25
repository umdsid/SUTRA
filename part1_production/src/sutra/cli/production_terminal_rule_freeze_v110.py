from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import pandas as pd
import pyarrow as pa,pyarrow.parquet as pq

from sutra.hierarchy.v110.terminal_rule import (
    validate_frozen_config,canonical_json_hash,manifest_files
)

def req(p):
    p=Path(p)
    if not p.exists():raise FileNotFoundError(p)
    return p

def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--config",default="configs/hierarchy_v110_production_terminal_rule.json")
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    upstream=json.loads(req(
        project/"results"/"hierarchy_v10931_ordered_merge_shard_replay"/
        "ordered_merge_shard_replay_global_certificate.json"
    ).read_text())

    required_upstream=[
        "LINEAGE_MASS_CONSERVATION_CERTIFIED",
        "EXPRESSION_COHERENCE_READY",
        "SPATIAL_COHERENCE_READY",
        "TERMINAL_RULE_PREREQUISITES_READY",
        "READY_TO_FREEZE_PRODUCTION_TERMINAL_RULE"
    ]
    failed=[k for k in required_upstream if upstream.get(k) is not True]
    if failed:
        raise SystemExit(f"ERROR: v1.0.9.3.1 prerequisites failed: {failed}")

    cfg=json.loads(req(project/a.config).read_text())
    errs=validate_frozen_config(cfg)
    if errs:
        raise SystemExit("ERROR: invalid terminal-rule contract: "+"; ".join(errs))

    out=project/"results"/"hierarchy_v110_production_terminal_rule_freeze"
    out.mkdir(parents=True,exist_ok=True)

    frozen=out/"production_terminal_rule_FROZEN.json"
    frozen.write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n")

    patterns=[
        "configs/hierarchy_v093*.json",
        "configs/hierarchy_v094*.json",
        "configs/hierarchy_v0941*.json",
        "configs/hierarchy_v100*.json",
        "configs/hierarchy_v101*.json",
        "configs/hierarchy_v102*.json",
        "configs/hierarchy_v103*.json",
        "configs/hierarchy_v104*.json",
        "configs/hierarchy_v105*.json",
        "configs/hierarchy_v106*.json",
        "configs/hierarchy_v1061*.json",
        "configs/hierarchy_v107*.json",
        "configs/hierarchy_v108*.json",
        "configs/hierarchy_v109*.json",
        "configs/hierarchy_v1091*.json",
        "configs/hierarchy_v1092*.json",
        "configs/hierarchy_v1093*.json",
        "configs/hierarchy_v10931*.json",
        "src/sutra.hierarchy/v093/*.py",
        "src/sutra.hierarchy/v094/*.py",
        "src/sutra.hierarchy/v0941/*.py",
        "src/sutra.hierarchy/v100/*.py",
        "src/sutra.hierarchy/v101/*.py",
        "src/sutra.hierarchy/v102/*.py",
        "src/sutra.hierarchy/v103/*.py",
        "src/sutra.hierarchy/v104/*.py",
        "src/sutra.hierarchy/v105/*.py",
        "src/sutra.hierarchy/v106/*.py",
        "src/sutra.hierarchy/v1061/*.py",
        "src/sutra.hierarchy/v107/*.py",
        "src/sutra.hierarchy/v108/*.py",
        "src/sutra.hierarchy/v109/*.py",
        "src/sutra.hierarchy/v1091/*.py",
        "src/sutra.hierarchy/v1092/*.py",
        "src/sutra.hierarchy/v1093/*.py",
        "src/sutra.hierarchy/v10931/*.py",
    ]
    manifest=manifest_files(project,patterns)
    wpq(manifest,out/"scientific_source_hash_manifest.parquet")
    manifest.to_csv(out/"scientific_source_hash_manifest.csv",index=False)

    contract_hash=canonical_json_hash(cfg)
    cert={
        "strata_version":"1.0.10",
        "stage":"production terminal-rule freeze",
        "upstream_reconstruction_certified":True,
        "terminal_rule_contract_sha256":contract_hash,
        "terminal_rule_frozen":True,
        "target_node_count_present":False,
        "tessellation_guard_is_target":False,
        "guard_can_trigger_success":False,
        "causal_trailing_windows_only":True,
        "future_fill_allowed":False,
        "centered_smoothing_allowed":False,
        "persistent_confirmation_required":True,
        "mass_stability_required":True,
        "expression_stability_required":True,
        "spatial_stability_required":True,
        "required_scientific_blocks":cfg["scientific_blocks"],
        "required_checkpoint_payload":cfg["required_checkpoint_payload"],
        "source_files_hashed":int(len(manifest)),
        "PRODUCTION_TERMINAL_RULE_FREEZE_GATE":"PASS",
        "PRODUCTION_TERMINAL_RULE_FROZEN":True,
        "READY_FOR_FINAL_PRODUCTION_RERUN":True,
        "SCIENTIFIC_HIERARCHY_CHANGED":False,
        "NEW_COARSE_GRAINING_PERFORMED":False
    }
    p=out/"production_terminal_rule_freeze_global_certificate.json"
    p.write_text(json.dumps(cert,indent=2)+"\n")

    print("STRATA 1.0.10 | Production terminal-rule freeze")
    print("No new coarse-graining.")
    print("Exact v1.0.9.3.1 lineage/expression/spatial prerequisites verified: PASS")
    print("Prospective detector uses trailing causal windows only.")
    print("No target node count exists.")
    print("Minimum node count is a catastrophic tessellation guard only and cannot produce success.")
    print("Persistent multiblock slowdown + mass stability + expression stability + spatial stability are required.")
    print("Production checkpoints must persist exact labels, ancestry, full state/provenance, and first/second scale differences.")
    print(f"Scientific source/config files hashed: {len(manifest)}")
    print(f"Frozen terminal-rule SHA256: {contract_hash}")
    print()
    print("PRODUCTION TERMINAL RULE FREEZE GATE: PASS")
    print("PRODUCTION TERMINAL RULE FROZEN: True")
    print("READY FOR FINAL PRODUCTION RERUN: True")
    print("SCIENTIFIC HIERARCHY CHANGED: False")
    print(f"Certificate: {p}")

if __name__=="__main__":
    main()
