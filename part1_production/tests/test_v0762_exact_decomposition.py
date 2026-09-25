import numpy as np,pandas as pd
from strata_hierarchy.v076.pressure_field_finalize import decompose_pressure_field

def test_observed_equals_potential_plus_residual():
    e=pd.DataFrame({
        "cell_i_index":[0,1,0],
        "cell_j_index":[1,2,2],
        "delta_pressure_z":[1.,1.,5.],
        "mechanics_delta_p_valid":[True,True,True],
    })
    c,E,s=decompose_pressure_field(e,3)
    assert np.allclose(
        E.delta_p_observed,
        E.delta_p_potential+E.delta_p_residual,
        atol=1e-12
    )
