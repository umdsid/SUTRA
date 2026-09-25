"""Immutable provenance record for a frozen Level-0 hierarchy object."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Provenance:
    geometry_certificate_sha256: str
    observability_certificate_sha256: str
    mechanics_certificate_sha256: str
    hierarchy_version: str


__all__ = ["Provenance"]
