from strata_hierarchy import CellID, PatchID, GraphID, InterfaceID, Interface
from strata_hierarchy.builders import build_graph


def test_graph_builder_linear_shape_smoke():
    n = 2000
    cells = tuple(CellID(i) for i in range(n))
    edges = tuple(
        Interface(
            id=InterfaceID(i),
            patch_id=PatchID(0),
            cell_i=CellID(i),
            cell_j=CellID(i + 1),
            length=1.0,
            curvature=0.0,
            owner=CellID(i),
            tension=None,
            delta_pressure=None,
        )
        for i in range(n - 1)
    )
    g = build_graph(GraphID(0), cells, edges)
    assert g.n_nodes == n
    assert g.n_undirected_edges == n - 1
    assert len(g.indptr) == n + 1
    assert len(g.indices) == 2 * (n - 1)
