"""Compact immutable CSR graph container."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .identifiers import GraphID, CellID


@dataclass(frozen=True, slots=True)
class Graph:
    """
    Cell-cell adjacency in canonical CSR form.

    Arrays are marked read-only by the builder before construction.
    """

    id: GraphID
    n_nodes: int
    indptr: np.ndarray
    indices: np.ndarray
    interface_ids: np.ndarray
    node_ids: tuple[CellID, ...]

    @property
    def n_directed_edges(self) -> int:
        return int(self.indices.size)

    @property
    def n_undirected_edges(self) -> int:
        return int(self.indices.size // 2)


__all__ = ["Graph"]
