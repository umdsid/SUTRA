from strata_native_mechanics.synthetic import three_cell_with_hole_mask
from strata_native_mechanics.geometry_fast import FastGeometryConfig,extract_interfaces_fast,extract_junctions_fast

def test_fast_junction_extraction_runs():
    m=three_cell_with_hole_mask()
    E,B,bg=extract_interfaces_fast(m,FastGeometryConfig(min_interface_pixels=2))
    J=extract_junctions_fast(m,E,bg)
    assert J is not None
