"""Strict builder-side validation for immutable hierarchy objects."""

from __future__ import annotations

import numpy as np

from ..biology import Cell, Patch
from ..interfaces import Interface
from ..graph import Graph


class HierarchyValidationError(ValueError):
    """Raised when a Level-0 invariant is violated."""


def _unique_ints(values, what: str) -> None:
    a = np.fromiter((int(x) for x in values), dtype=np.int64)
    if a.size and np.unique(a).size != a.size:
        raise HierarchyValidationError(f"duplicate {what}")


def validate_cells(cells: tuple[Cell, ...]) -> None:
    _unique_ints((c.id for c in cells), "cell IDs")
    for c in cells:
        if not np.isfinite(c.centroid_x) or not np.isfinite(c.centroid_y):
            raise HierarchyValidationError(f"cell {int(c.id)} has non-finite centroid")
        if not np.isfinite(c.area) or c.area <= 0:
            raise HierarchyValidationError(f"cell {int(c.id)} has invalid area")
        _unique_ints(c.interface_ids, f"interface IDs on cell {int(c.id)}")


def validate_interfaces(
    interfaces: tuple[Interface, ...],
    valid_cell_ids: set[int],
) -> None:
    _unique_ints((e.id for e in interfaces), "interface IDs")
    seen_pairs: set[tuple[int, int]] = set()

    for e in interfaces:
        i = int(e.cell_i)
        if i not in valid_cell_ids:
            raise HierarchyValidationError(
                f"interface {int(e.id)} references missing cell_i={i}"
            )

        if e.cell_j is not None:
            j = int(e.cell_j)
            if j not in valid_cell_ids:
                raise HierarchyValidationError(
                    f"interface {int(e.id)} references missing cell_j={j}"
                )
            if i == j:
                raise HierarchyValidationError(
                    f"interface {int(e.id)} is a self-interface"
                )
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                raise HierarchyValidationError(
                    f"duplicate cell-cell pair {pair}"
                )
            seen_pairs.add(pair)

        if int(e.owner) not in valid_cell_ids:
            raise HierarchyValidationError(
                f"interface {int(e.id)} has missing owner={int(e.owner)}"
            )
        if not np.isfinite(e.length) or e.length <= 0:
            raise HierarchyValidationError(
                f"interface {int(e.id)} has invalid length"
            )


def validate_graph(graph: Graph) -> None:
    indptr = np.asarray(graph.indptr)
    indices = np.asarray(graph.indices)
    eids = np.asarray(graph.interface_ids)

    if graph.n_nodes != len(graph.node_ids):
        raise HierarchyValidationError("graph n_nodes != len(node_ids)")
    if indptr.ndim != 1 or indices.ndim != 1 or eids.ndim != 1:
        raise HierarchyValidationError("CSR arrays must be one-dimensional")
    if len(indptr) != graph.n_nodes + 1:
        raise HierarchyValidationError("CSR indptr has wrong length")
    if len(indices) != len(eids):
        raise HierarchyValidationError("graph indices/interface_ids length mismatch")
    if indptr[0] != 0 or indptr[-1] != len(indices):
        raise HierarchyValidationError("invalid CSR indptr endpoints")
    if np.any(indptr[1:] < indptr[:-1]):
        raise HierarchyValidationError("CSR indptr is not monotone")
    if len(indices) and (indices.min() < 0 or indices.max() >= graph.n_nodes):
        raise HierarchyValidationError("CSR neighbor index out of range")

    node_pos = {int(cid): k for k, cid in enumerate(graph.node_ids)}
    if len(node_pos) != graph.n_nodes:
        raise HierarchyValidationError("duplicate graph node IDs")

    # Exact undirected symmetry including interface identity.
    directed = set()
    for u in range(graph.n_nodes):
        for k in range(indptr[u], indptr[u + 1]):
            directed.add((u, int(indices[k]), int(eids[k])))
    for u, v, eid in directed:
        if (v, u, eid) not in directed:
            raise HierarchyValidationError(
                f"graph adjacency missing reciprocal edge for interface {eid}"
            )


def validate_patch(patch: Patch) -> None:
    if patch.n_cells != len(patch.cell_ids):
        raise HierarchyValidationError(
            f"patch {int(patch.id)} n_cells mismatch"
        )
    if patch.n_interfaces != len(patch.interface_ids):
        raise HierarchyValidationError(
            f"patch {int(patch.id)} n_interfaces mismatch"
        )
    _unique_ints(patch.cell_ids, f"cell IDs in patch {int(patch.id)}")
    _unique_ints(
        patch.interface_ids, f"interface IDs in patch {int(patch.id)}"
    )


__all__ = [
    "HierarchyValidationError",
    "validate_cells",
    "validate_interfaces",
    "validate_graph",
    "validate_patch",
]
