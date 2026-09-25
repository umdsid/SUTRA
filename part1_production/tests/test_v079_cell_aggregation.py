import numpy as np,pandas as pd
from sutra.hierarchy.v079.curvature_summary import cell_aggregate

def test_loop_quantity_attaches_to_all_incident_cells():
    x=pd.DataFrame({
        "loop_nodes":["0;1;2"],
        "fully_resolved":[True],
        "directional_resolved_loop":[True],
        "density_defined":[True],
        "spectral_angle_rms":[1.2],
        "spatial_area":[2.0],
        "holonomy_density_rms":[.6],
    })
    y=cell_aggregate(x,4)
    assert (y.loc[:2,"incident_directional_loops"]==1).all()
    assert y.loc[3,"incident_directional_loops"]==0
    assert np.allclose(y.loc[:2,"holonomy_angle_rms_sum"],1.2)
