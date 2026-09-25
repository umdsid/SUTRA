from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import pandas as pd

from sutra.hierarchy.v071.resource_intake import (
    locate_functional_resources,
    locate_cellchat_interaction,
    canonicalize_cellchat,
    copy_functional_snapshot,
    sha256_file,
)


def split_env(name):
    x=os.environ.get(name,"").strip()
    if not x:
        return []
    return [Path(p) for p in x.split(os.pathsep) if p.strip()]


def existing_search_roots(project:Path,args):
    roots=[project/"resources"]
    roots.extend(Path(x) for x in (args.search_root or []))
    roots.extend(split_env("STRATA_RESOURCE_SEARCH_ROOTS"))
    roots.extend(split_env("STRATA_FUNCTIONAL_ROOTS"))
    roots.extend(split_env("STRATA_SIGNALING_ROOTS"))
    # Explicit source arguments can be either a file or directory.
    if args.functional_root:
        roots.extend(Path(x) for x in args.functional_root)
    if args.signaling_root:
        roots.extend(Path(x) for x in args.signaling_root)
    out=[];seen=set()
    for p in roots:
        q=Path(p).expanduser()
        if q.exists():
            q=q.resolve()
            if q not in seen:
                seen.add(q);out.append(q)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--search-root",action="append")
    ap.add_argument("--functional-root",action="append")
    ap.add_argument("--signaling-root",action="append")
    ap.add_argument(
        "--no-spotlight",action="store_true",
        help="Reserved compatibility flag; explicit roots remain preferred."
    )
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    freeze=project/"results"/"hierarchy_level0_v070"/"level0_freeze_certificate.json"
    if not freeze.exists():
        raise SystemExit("ERROR: v0.7.0 freeze certificate missing")
    fd=json.loads(freeze.read_text())
    if fd.get("LEVEL0_FREEZE")!="PASS":
        raise SystemExit("ERROR: v0.7.0 Level-0 freeze is not PASS")

    roots=existing_search_roots(project,a)
    snapshot=project/"resources"/"strata_hierarchy_v0711"
    functional_dest=snapshot/"functional"
    signaling_dest=snapshot/"signaling"
    functional_dest.mkdir(parents=True,exist_ok=True)
    signaling_dest.mkdir(parents=True,exist_ok=True)

    print("STRATA 0.7.1.1 | Pinned hierarchy resource intake")
    print("Frozen Level-0 remains read-only.")
    print(f"Explicit/local search roots: {len(roots)}")
    for r in roots:
        print(f"  - {r}")

    # Functional resources
    chosen=locate_functional_resources(roots)
    copied,functional_manifest=copy_functional_snapshot(
        chosen,functional_dest
    )
    print(f"\nFunctional GMT resources snapshot: {len(copied)}")
    for p in copied:
        print(f"  [GMT] {p.name}")

    # CellChat native resource
    cc=locate_cellchat_interaction(roots)
    signaling_manifest=[]
    canonical=pd.DataFrame()
    cellchat_audit={
        "status":"HOLD",
        "reason":"CellChat interaction_input_CellChatDB.csv not found",
    }
    if cc is not None:
        canonical,counts=canonicalize_cellchat(cc)
        src_copy=signaling_dest/"interaction_input_CellChatDB.source.csv"
        shutil.copy2(cc,src_copy)
        canonical_path=signaling_dest/"cellchat_typed_interactions.csv"
        canonical.to_csv(canonical_path,index=False)
        cellchat_audit={
            "status":"PASS" if len(canonical)>0 else "HOLD",
            "source_path":str(cc),
            "source_sha256":sha256_file(cc),
            "snapshot_source_path":str(src_copy),
            "snapshot_source_sha256":sha256_file(src_copy),
            "canonical_path":str(canonical_path),
            "canonical_sha256":sha256_file(canonical_path),
            **counts,
        }
        signaling_manifest.append(cellchat_audit)
        print(
            f"\nCellChat: source rows={counts['n_source_rows']:,} "
            f"typed simple interactions={counts['n_emitted_rows']:,}"
        )
        print(
            "  unsupported annotation rows="
            f"{counts['n_unsupported_annotation']:,} | "
            "complex/non-simple rows="
            f"{counts['n_complex_or_nonsimple_entity']:,}"
        )
    else:
        print("\nCellChat: NOT FOUND")

    functional_ready=len(copied)>0
    signaling_ready=cellchat_audit.get("status")=="PASS"
    status="PASS" if functional_ready and signaling_ready else "HOLD"

    manifest={
        "strata_version":"0.7.1.1",
        "stage":"pinned hierarchy resource intake",
        "level0_freeze_sha256":sha256_file(freeze),
        "search_roots":[str(r) for r in roots],
        "functional":{
            "status":"PASS" if functional_ready else "HOLD",
            "resources":functional_manifest,
        },
        "signaling":{
            "status":"PASS" if signaling_ready else "HOLD",
            "cellchat":cellchat_audit,
        },
        "resource_role_separation":{
            "functional_gmts_used_only_for_molecular_prior":True,
            "cellchat_used_only_for_directed_communication":True,
        },
        "RESOURCE_INTAKE_GATE":status,
    }
    p=snapshot/"resource_intake_manifest.json"
    p.write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"\nRESOURCE INTAKE GATE: {status}")
    print(f"Manifest: {p}")

    if status=="HOLD":
        print("\nIf your resources live elsewhere, rerun with e.g.:")
        print(
            "  ./run_hierarchy_v0711_resources.sh "
            "/path/to/msigdb /path/to/CellChat"
        )
    return 0


if __name__=="__main__":
    raise SystemExit(main())
