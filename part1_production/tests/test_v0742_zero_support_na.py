import numpy as np
from strata_hierarchy.v074.directional_comm import directional_state


def test_zero_support_directionality_is_na():
    d=directional_state([0.],[0.],0.1).iloc[0]
    assert not bool(d.comm_supported)
    assert np.isnan(d.comm_directionality)
    assert np.isnan(d.comm_asymmetry)
