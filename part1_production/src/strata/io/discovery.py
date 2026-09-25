from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class XeniumSampleAssets:
    """Resolved measured assets for one STRATA Xenium specimen."""

    sample_id: str
    root: Path
    matrix: Path
    cells: Path | None
    cell_boundaries: Path
    nucleus_boundaries: Path | None
    transcripts: Path | None

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "root": str(self.root),
            "matrix": str(self.matrix),
            "cells": None if self.cells is None else str(self.cells),
            "cell_boundaries": str(self.cell_boundaries),
            "nucleus_boundaries": (
                None if self.nucleus_boundaries is None
                else str(self.nucleus_boundaries)
            ),
            "transcripts": (
                None if self.transcripts is None else str(self.transcripts)
            ),
        }


def _hits(root: Path, patterns: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for p in root.rglob(pattern):
            if p.is_file():
                q = p.resolve()
                if q not in seen:
                    seen.add(q)
                    found.append(q)
    return sorted(found)


def _one(
    root: Path,
    patterns: Iterable[str],
    *,
    required: bool,
    asset_name: str,
) -> Path | None:
    hits = _hits(root, patterns)
    if not hits:
        if required:
            raise FileNotFoundError(
                f"{root.name}: required Xenium asset '{asset_name}' not found; "
                f"patterns={tuple(patterns)}"
            )
        return None

    # A specimen directory should resolve to exactly one canonical measured
    # asset of each type.  Silent first-match behavior is unsafe when stale
    # copies or derived outputs are present.
    if len(hits) > 1:
        raise RuntimeError(
            f"{root.name}: ambiguous Xenium asset '{asset_name}': "
            + ", ".join(str(x) for x in hits)
        )
    return hits[0]


def discover_xenium_samples(
    data_root: str | Path,
    require_cells: bool = True,
) -> list[XeniumSampleAssets]:
    """
    Discover measured Xenium specimens beneath STRATA/data.

    Discovery is specimen-directory scoped and deterministic.  It recognizes
    both canonical 10x names and the normalized names used by STRATA.

    Required assets:
      - cell-feature matrix H5
      - cell boundaries
      - cells table when require_cells=True

    Optional:
      - nucleus boundaries
      - transcripts
    """
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    samples: list[XeniumSampleAssets] = []
    for sroot in sorted(p for p in root.iterdir() if p.is_dir()):
        matrix_hits = _hits(
            sroot,
            (
                "*cell_feature_matrix*.h5",
                "*feature_matrix*.h5",
            ),
        )
        boundary_hits = _hits(
            sroot,
            (
                "*cell_boundaries*.parquet",
                "*cell_boundaries*.parquet.gz",
            ),
        )

        # Non-specimen helper/resource directories are ignored.
        if not matrix_hits and not boundary_hits:
            continue

        matrix = _one(
            sroot,
            ("*cell_feature_matrix*.h5", "*feature_matrix*.h5"),
            required=True,
            asset_name="matrix",
        )
        cell_boundaries = _one(
            sroot,
            ("*cell_boundaries*.parquet", "*cell_boundaries*.parquet.gz"),
            required=True,
            asset_name="cell_boundaries",
        )
        cells = _one(
            sroot,
            ("*cells.parquet", "*cells.parquet.gz", "*cells.csv"),
            required=require_cells,
            asset_name="cells",
        )
        nucleus_boundaries = _one(
            sroot,
            (
                "*nucleus_boundaries*.parquet",
                "*nucleus_boundaries*.parquet.gz",
            ),
            required=False,
            asset_name="nucleus_boundaries",
        )
        transcripts = _one(
            sroot,
            (
                "*transcripts*.parquet",
                "*transcripts*.parquet.gz",
            ),
            required=False,
            asset_name="transcripts",
        )

        samples.append(
            XeniumSampleAssets(
                sample_id=sroot.name,
                root=sroot,
                matrix=matrix,
                cells=cells,
                cell_boundaries=cell_boundaries,
                nucleus_boundaries=nucleus_boundaries,
                transcripts=transcripts,
            )
        )

    return samples


def discover_samples(data_root):
    """
    Backward-compatible legacy discovery API.

    Older Tranche 2 callers require dictionaries containing only sample/root/
    cell_boundaries.  Preserve that interface while routing through the same
    deterministic specimen-directory scan where possible.
    """
    root = Path(data_root)
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        b = _hits(
            d,
            (
                "*cell_boundaries*.parquet",
                "*cell_boundaries*.parquet.gz",
            ),
        )
        if not b:
            continue
        out.append({"sample": d.name, "root": d, "cell_boundaries": b[0]})
    return out


__all__ = [
    "XeniumSampleAssets",
    "discover_xenium_samples",
    "discover_samples",
]
