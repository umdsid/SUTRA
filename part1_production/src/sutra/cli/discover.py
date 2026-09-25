from __future__ import annotations

import argparse
import json
from pathlib import Path

from sutra.io.discovery import discover_xenium_samples


def main(argv=None):
    p = argparse.ArgumentParser(description="Discover measured Xenium datasets for STRATA.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--output", default="manifests/generated_xenium_manifest.json")
    args = p.parse_args(argv)

    samples = discover_xenium_samples(args.data_root)
    payload = {"schema_version": "sutra.dataset_manifest.v1", "samples": [s.to_dict() for s in samples]}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"STRATA discovery: {len(samples)} sample(s)")
    for s in samples:
        print(f"  {s.sample_id}: {s.root}")
    print(f"Manifest: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
