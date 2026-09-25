import numpy as np,pandas as pd
from strata_hierarchy.v076.pressure_field_finalize import decompose_pressure_field

def test_transport_edge_component_is_gauge_invariant():
    e=pd.DataFrame({
        "cell_i_index":[0,1],
        "cell_j_index":[1,2],
        "delta_pressure_z":[1.,1.],
        "mechanics_delta_p_valid":[True,True],
    })
    c,E,s=decompose_pressure_field(e,3)
    p=c.pressure_potential.to_numpy()
    shifted=p+123.456
    assert np.allclose(
        shifted[E.cell_i_index.to_numpy()]-shifted[E.cell_j_index.to_numpy()],
        E.delta_p_potential
    )
