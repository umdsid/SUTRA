import numpy as np
from strata_native_mechanics.synthetic import three_cell_with_hole_mask
from strata_native_mechanics.geometry import GeometryConfig,extract_interfaces
from strata_native_mechanics.geometry_fast import FastGeometryConfig,extract_interfaces_fast

def test_fast_interface_pairs_match_reference():
    m=three_cell_with_hole_mask()
    a,_,_=extract_interfaces(m,GeometryConfig(min_interface_pixels=2))
    b,_,_=extract_interfaces_fast(m,FastGeometryConfig(min_interface_pixels=2))
    pa=set()
    pb=set()
    for r in a.itertuples():
        pa.add((r.kind,int(r.cell_i),None if r.cell_j is None or (isinstance(r.cell_j,float) and np.isnan(r.cell_j)) else int(r.cell_j),
                None if r.background_component is None or (isinstance(r.background_component,float) and np.isnan(r.background_component)) else int(r.background_component)))
    for r in b.itertuples():
        pb.add((r.kind,int(r.cell_i),None if r.cell_j is None or (isinstance(r.cell_j,float) and np.isnan(r.cell_j)) else int(r.cell_j),
                None if r.background_component is None or (isinstance(r.background_component,float) and np.isnan(r.background_component)) else int(r.background_component)))
    assert pa==pb
