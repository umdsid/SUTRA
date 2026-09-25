"""Top-level immutable tissue sample descriptor."""

from __future__ import annotations

from dataclasses import dataclass

from .identifiers import SampleID, PatchID
from .provenance import Provenance


@dataclass(frozen=True, slots=True)
class Sample:
    id: SampleID
    name: str
    patch_ids: tuple[PatchID, ...]
    provenance: Provenance


__all__ = ["Sample"]
