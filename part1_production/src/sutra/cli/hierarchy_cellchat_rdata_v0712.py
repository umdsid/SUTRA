from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from sutra.hierarchy.v071.cellchat_rdata import (
    locate_cellchat_database,
    export_cellchat_interaction,
)
from sutra.hierarchy.v071.resource_intake import (
    canonicalize_cellchat,
    sha256_file,
)


def _split_env(name):
    x = os.environ.get(name, "").strip()
    return [Path(p) for p in x.split(os.pathsep) if p.strip()] if x else []


def _roots(project: Path, explicit):
    xs = [project / "resources"]
    xs.extend(Path(x) for x in explicit)
    xs.extend(_split_env("STRATA_RESOURCE_SEARCH_ROOTS"))
    xs.extend(_split_env("STRATA_SIGNALING_ROOTS"))
    out = []
    seen = set()
    for p in xs:
        q = Path(p).expanduser()
        if q.exists():
            q = q.resolve()
            if q not in seen:
                seen.add(q)
                out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--search-root", action="append", default=[])
    a = ap.parse_args()

    project = Path(a.project_root).resolve()
    freeze = (
        project / "results" / "hierarchy_level0_v070" /
        "level0_freeze_certificate.json"
    )
    if not freeze.exists():
        raise SystemExit("ERROR: frozen v0.7.0 certificate missing")
    fd = json.loads(freeze.read_text())
    if fd.get("LEVEL0_FREEZE") != "PASS":
        raise SystemExit("ERROR: frozen v0.7.0 is not PASS")

    roots = _roots(project, a.search_root)
    print("STRATA 0.7.1.2 | CellChat native R-data intake")
    print("No hierarchy or Level-0 mathematics are changed.")
    for r in roots:
        print(f"  search: {r}")

    found = locate_cellchat_database(roots)
    dest = project / "resources" / "strata_hierarchy_v0711" / "signaling"
    dest.mkdir(parents=True, exist_ok=True)

    if found is None:
        report = {
            "status": "HOLD",
            "reason": "No CellChat interaction CSV or CellChatDB.human R-data object found",
            "search_roots": [str(x) for x in roots],
        }
        (dest / "cellchat_rdata_intake.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print("\nCELLCHAT R-DATA INTAKE: HOLD")
        return 0

    kind, source = found
    print(f"\nFound CellChat source: {kind}")
    print(f"  {source}")

    export_csv = dest / "interaction_input_CellChatDB.exported.csv"

    if kind == "interaction_csv":
        shutil.copy2(source, export_csv)
        source_meta = {
            "source_kind": kind,
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "export_path": str(export_csv),
            "export_sha256": sha256_file(export_csv),
            "export_method": "byte_copy",
        }
    else:
        extractor = project / "scripts" / "export_cellchat_interaction_v0712.R"
        source_meta = export_cellchat_interaction(
            source, kind, export_csv, extractor
        )

    canonical, audit = canonicalize_cellchat(export_csv)
    canonical_path = dest / "cellchat_typed_interactions.csv"
    canonical.to_csv(canonical_path, index=False)

    report = {
        "status": "PASS" if len(canonical) else "HOLD",
        "source": source_meta,
        "canonical_path": str(canonical_path),
        "canonical_sha256": sha256_file(canonical_path),
        "canonicalization": audit,
        "typing_rule": {
            "Cell-Cell Contact": "contact",
            "Secreted Signaling": "diffusible",
            "ECM-Receptor": "audit_only",
            "complex_entities": "not exploded",
        },
    }
    (dest / "cellchat_rdata_intake.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print(f"\nTyped canonical interactions: {len(canonical):,}")
    print(f"CELLCHAT R-DATA INTAKE: {report['status']}")
    print(f"Registry: {canonical_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
