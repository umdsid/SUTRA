import numpy as np,pandas as pd
from strata_hierarchy.v076.geometry_hierarchy import reconstruct_level0_mechanics

def test_pressure_difference_reconstruction():
    # true p = [1,0,-1], hence dp_ij=p_i-p_j
    e=pd.DataFrame({
        "cell_i_index":[0,1],
        "cell_j_index":[1,2],
        "tension_z":[.2,.3],
        "delta_pressure_z":[1.,1.],
        "mechanics_tension_valid":[True,True],
        "mechanics_delta_p_valid":[True,True],
    })
    s,a=reconstruct_level0_mechanics(e,3)
    p=s.pressure_state.to_numpy()
    assert np.allclose([p[0]-p[1],p[1]-p[2]],[1.,1.],atol=1e-7)
    assert a["residual_rms"]<1e-7
