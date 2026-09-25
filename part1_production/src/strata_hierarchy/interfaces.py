"""Immutable canonical tissue-interface object."""

from __future__ import annotations

from dataclasses import dataclass

from .identifiers import InterfaceID, CellID, PatchID


@dataclass(frozen=True, slots=True)
class Interface:
    """
    Canonical Level-0 interface.

    Mechanical values are optional because unresolved/non-certified quantities
    are intentionally absent from production mechanics.
    """

    id: InterfaceID
    patch_id: PatchID
    cell_i: CellID
    cell_j: CellID | None
    length: float
    curvature: float
    owner: CellID
    tension: float | None
    delta_pressure: float | None
    flags: int = 0


__all__ = ["Interface"]
