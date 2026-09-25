import pandas as pd
from strata_native_mechanics.synthetic import three_cell_with_hole_mask
from strata_native_mechanics.geometry_fast import (
    FastGeometryConfig,extract_interfaces_fast,extract_junctions_fast,extract_cell_centroids_fast
)
from strata_native_mechanics.nullspace_forensics import patch_features

def test_patch_feature_vector():
    m=three_cell_with_hole_mask()
    E,B,bg=extract_interfaces_fast(m,FastGeometryConfig(min_interface_pixels=2))
    J=extract_junctions_fast(m,E,bg)
    C=extract_cell_centroids_fast(m)
    f=patch_features([1,2,3],E,B,J,C)
    assert f["n_patch_cells"]==3
    assert "graph_mean_degree" in f
    assert "low_information_curvature_fraction" in f
    assert "junctions_per_cell" in f
    assert "cell_boundary_interface_fraction" in f
