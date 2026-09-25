"""Immutable biological objects used by the STRATA hierarchy."""

from __future__ import annotations

from dataclasses import dataclass

from .identifiers import CellID, PatchID, GraphID, InterfaceID


@dataclass(frozen=True, slots=True)
class Cell:
    """Immutable Level-0 biological cell reference object."""

    id: CellID
    patch_id: PatchID
    centroid_x: float
    centroid_y: float
    area: float
    gene_index: int
    interface_ids: tuple[InterfaceID, ...]


@dataclass(frozen=True, slots=True)
class Patch:
    """Immutable patch consisting only of object references."""

    id: PatchID
    graph_id: GraphID
    cell_ids: tuple[CellID, ...]
    interface_ids: tuple[InterfaceID, ...]
    n_cells: int
    n_interfaces: int


__all__ = ["Cell", "Patch"]
