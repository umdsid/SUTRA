import numpy as np,pandas as pd
from sutra.hierarchy.v079.curvature_summary import (
    polygon_geometry,attach_holonomy_density
)

def test_collinear_loop_is_not_divided_by_zero_area():
    g=polygon_geometry(np.array([[0.,0.],[1.,0.],[2.,0.]]))
    assert not g["spatial_nondegenerate"]
    x=pd.DataFrame({
        "fully_resolved":[True],
        "spatial_nondegenerate":[False],
        "spatial_area":[g["spatial_area"]],
        "spectral_angle_rms":[1.],
        "spectral_angle_max":[1.],
        "identity_deviation_normalized":[1.],
        "n_directional_edges":[1],
    })
    y=attach_holonomy_density(x)
    assert np.isnan(y.loc[0,"holonomy_density_rms"])
    assert not y.loc[0,"density_defined"]
