import pandas as pd
from shapely.geometry import Polygon
from strata.io.boundaries import polygons_from_boundary_table
from strata.topology.interfaces import (
    build_observed_interfaces,
    TopologyConfig,
    _safe_geom,
)
from strata.topology.diagnostics import reciprocity_audit


def square(cid, x0, y0, x1, y1):
    return pd.DataFrame({
        "cell_id":[cid]*4,
        "x":[x0,x1,x1,x0],
        "y":[y0,y0,y1,y1],
    })


def test_exact_shared_interface():
    df = pd.concat([square("A",0,0,1,1), square("B",1,0,2,1)], ignore_index=True)
    polys, _ = polygons_from_boundary_table(df)
    edges, nodes = build_observed_interfaces(
        polys, TopologyConfig(min_shared_length=0.5, contact_tolerance=0.0)
    )
    assert len(edges) == 1
    assert abs(edges.iloc[0].shared_length - 1.0) < 1e-8
    assert set(nodes.degree) == {1}
    assert reciprocity_audit(edges)["status"] == "PASS"


def test_non_contact_not_created():
    df = pd.concat([square("A",0,0,1,1), square("B",2,0,3,1)], ignore_index=True)
    polys, _ = polygons_from_boundary_table(df)
    edges, nodes = build_observed_interfaces(
        polys, TopologyConfig(min_shared_length=0.5, contact_tolerance=0.1)
    )
    assert len(edges) == 0
    assert int(nodes.degree.sum()) == 0


def test_none_geometry_is_skipped():
    polys = {
        "A": Polygon([(0,0),(1,0),(1,1),(0,1)]),
        "B": None,
        "C": Polygon([(1,0),(2,0),(2,1),(1,1)]),
    }
    edges, nodes = build_observed_interfaces(
        polys, TopologyConfig(min_shared_length=0.5, contact_tolerance=0.0)
    )
    assert len(edges) == 1
    assert set(nodes.cell_id) == {"A","C"}


def test_safe_geom_normalizes_none():
    assert _safe_geom(None).is_empty
