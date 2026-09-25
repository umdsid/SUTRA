"""Linear-time CSR graph construction from canonical interfaces."""

from __future__ import annotations

import numpy as np

from ..graph import Graph
from ..identifiers import GraphID, CellID
from ..interfaces import Interface
from .validation import validate_graph


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


def build_graph(
    graph_id: GraphID,
    cell_ids: tuple[CellID, ...],
    interfaces: tuple[Interface, ...],
) -> Graph:
    """
    Build a deterministic undirected CSR adjacency.

    Background interfaces are intentionally absent from cell-cell adjacency.
    No NetworkX object and no scipy sparse intermediate is created.
    Complexity is O(N + M log d) from deterministic per-row sorting.
    """
    n = len(cell_ids)
    if n == 0:
        graph = Graph(
            id=graph_id,
            n_nodes=0,
            indptr=_readonly(np.zeros(1, dtype=np.int64)),
            indices=_readonly(np.empty(0, dtype=np.int32)),
            interface_ids=_readonly(np.empty(0, dtype=np.int64)),
            node_ids=(),
        )
        validate_graph(graph)
        return graph

    cell_to_pos = {int(cid): pos for pos, cid in enumerate(cell_ids)}
    if len(cell_to_pos) != n:
        raise ValueError("cell_ids contains duplicates")

    degree = np.zeros(n, dtype=np.int64)
    edge_records: list[tuple[int, int, int]] = []

    for e in interfaces:
        if e.cell_j is None:
            continue
        ci = int(e.cell_i)
        cj = int(e.cell_j)
        try:
            u = cell_to_pos[ci]
            v = cell_to_pos[cj]
        except KeyError as exc:
            raise ValueError(
                f"interface {int(e.id)} references cell outside graph"
            ) from exc
        if u == v:
            raise ValueError(f"interface {int(e.id)} creates self-edge")
        eid = int(e.id)
        edge_records.append((u, v, eid))
        degree[u] += 1
        degree[v] += 1

    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])

    nnz = int(indptr[-1])
    # int32 is enough for current sample sizes and halves adjacency memory.
    indices = np.empty(nnz, dtype=np.int32)
    interface_ids = np.empty(nnz, dtype=np.int64)
    cursor = indptr[:-1].copy()

    for u, v, eid in edge_records:
        ku = int(cursor[u])
        indices[ku] = v
        interface_ids[ku] = eid
        cursor[u] += 1

        kv = int(cursor[v])
        indices[kv] = u
        interface_ids[kv] = eid
        cursor[v] += 1

    # Deterministic row order by neighbor, then interface ID.
    for u in range(n):
        a, b = int(indptr[u]), int(indptr[u + 1])
        if b - a <= 1:
            continue
        order = np.lexsort((interface_ids[a:b], indices[a:b]))
        indices[a:b] = indices[a:b][order]
        interface_ids[a:b] = interface_ids[a:b][order]

    graph = Graph(
        id=graph_id,
        n_nodes=n,
        indptr=_readonly(indptr),
        indices=_readonly(indices),
        interface_ids=_readonly(interface_ids),
        node_ids=cell_ids,
    )
    validate_graph(graph)
    return graph


__all__ = ["build_graph"]
