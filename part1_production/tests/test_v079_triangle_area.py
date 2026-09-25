import numpy as np
from strata_hierarchy.v079.curvature_summary import polygon_geometry

def test_triangle_area_and_perimeter():
    g=polygon_geometry(np.array([[0.,0.],[1.,0.],[0.,1.]]))
    assert np.isclose(g["spatial_area"],.5)
    assert np.isclose(g["spatial_perimeter"],2+np.sqrt(2))
    assert g["spatial_nondegenerate"]
