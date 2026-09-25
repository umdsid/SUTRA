from dataclasses import FrozenInstanceError
import numpy as np
import pytest

from strata_hierarchy import (
    Cell, Patch, Interface,
    CellID, PatchID, GraphID, InterfaceID,
)
from strata_hierarchy.builders import (
    build_graph, validate_cells, validate_interfaces,
    HierarchyValidationError,
)


def test_objects_are_frozen():
    c = Cell(
        id=CellID(1),
        patch_id=PatchID(0),
        centroid_x=1.0,
        centroid_y=2.0,
        area=3.0,
        gene_index=0,
        interface_ids=(),
    )
    with pytest.raises(FrozenInstanceError):
        c.area = 9.0


def test_strict_cell_validation():
    bad = (
        Cell(
            id=CellID(1), patch_id=PatchID(0),
            centroid_x=0.0, centroid_y=0.0,
            area=-1.0, gene_index=0, interface_ids=(),
        ),
    )
    with pytest.raises(HierarchyValidationError):
        validate_cells(bad)


def test_interface_validation_rejects_duplicate_pair():
    e = (
        Interface(
            id=InterfaceID(1), patch_id=PatchID(0),
            cell_i=CellID(10), cell_j=CellID(11),
            length=1.0, curvature=0.0, owner=CellID(10),
            tension=None, delta_pressure=None,
        ),
        Interface(
            id=InterfaceID(2), patch_id=PatchID(0),
            cell_i=CellID(11), cell_j=CellID(10),
            length=1.0, curvature=0.0, owner=CellID(10),
            tension=None, delta_pressure=None,
        ),
    )
    with pytest.raises(HierarchyValidationError):
        validate_interfaces(e, {10, 11})


def test_csr_graph_is_deterministic_symmetric_and_readonly():
    cell_ids = (CellID(30), CellID(10), CellID(20))
    e = (
        Interface(
            id=InterfaceID(8), patch_id=PatchID(0),
            cell_i=CellID(30), cell_j=CellID(20),
            length=2.0, curvature=0.0, owner=CellID(20),
            tension=None, delta_pressure=None,
        ),
        Interface(
            id=InterfaceID(4), patch_id=PatchID(0),
            cell_i=CellID(10), cell_j=CellID(20),
            length=1.0, curvature=0.0, owner=CellID(10),
            tension=None, delta_pressure=None,
        ),
        Interface(
            id=InterfaceID(99), patch_id=PatchID(0),
            cell_i=CellID(30), cell_j=None,
            length=3.0, curvature=0.0, owner=CellID(30),
            tension=None, delta_pressure=None,
        ),
    )
    g = build_graph(GraphID(5), cell_ids, e)

    assert g.n_nodes == 3
    assert g.n_undirected_edges == 2
    assert g.indices.flags.writeable is False
    assert g.indptr.flags.writeable is False
    assert g.interface_ids.flags.writeable is False

    # Node order is caller-defined and retained exactly.
    assert tuple(map(int, g.node_ids)) == (30, 10, 20)

    directed = set()
    for u in range(g.n_nodes):
        for k in range(g.indptr[u], g.indptr[u+1]):
            directed.add((u, int(g.indices[k]), int(g.interface_ids[k])))
    assert (0, 2, 8) in directed and (2, 0, 8) in directed
    assert (1, 2, 4) in directed and (2, 1, 4) in directed
    assert all(eid != 99 for _, _, eid in directed)
