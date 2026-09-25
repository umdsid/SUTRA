from strata_native_mechanics.synthetic import three_cell_with_hole_mask
from strata_native_mechanics.geometry import (
    GeometryConfig,extract_interfaces,extract_junctions,extract_cell_centroids
)

def test_boundary_aware_geometry():
    m=three_cell_with_hole_mask()
    E,bg,bgl=extract_interfaces(m,GeometryConfig(min_interface_pixels=2))
    J=extract_junctions(m,E,bgl)
    C=extract_cell_centroids(m)
    assert len(C)==3
    assert (bg.kind=="internal_gap").any()
    assert (E.kind=="cell_boundary").any()
    assert (E.kind=="cell_cell").any()
