import numpy as np,pandas as pd
from strata_hierarchy.v079.curvature_summary import attach_holonomy_density

def test_density_is_angle_over_area():
    x=pd.DataFrame({
        "fully_resolved":[True],
        "spatial_nondegenerate":[True],
        "spatial_area":[2.0],
        "spectral_angle_rms":[1.0],
        "spectral_angle_max":[1.5],
        "identity_deviation_normalized":[.4],
        "n_directional_edges":[2],
    })
    y=attach_holonomy_density(x)
    assert np.isclose(y.loc[0,"holonomy_density_rms"],.5)
    assert np.isclose(y.loc[0,"holonomy_density_max"],.75)
