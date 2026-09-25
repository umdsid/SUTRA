import pandas as pd
from strata_hierarchy.v079.curvature_summary import attach_holonomy_density,certify_summary

def test_identity_loop_has_zero_density():
    x=pd.DataFrame({
        "fully_resolved":[True],
        "spatial_nondegenerate":[True],
        "spatial_area":[1.],
        "spatial_perimeter":[3.],
        "spectral_angle_rms":[0.],
        "spectral_angle_max":[0.],
        "identity_deviation_normalized":[0.],
        "n_directional_edges":[0],
    })
    y=attach_holonomy_density(x)
    c=certify_summary(y)
    assert y.loc[0,"holonomy_density_rms"]==0.
    assert c["certificate_pass"]
