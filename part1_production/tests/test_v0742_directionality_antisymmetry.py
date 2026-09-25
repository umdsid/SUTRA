import numpy as np
from strata_hierarchy.v074.directional_comm import directional_state


def test_directionality_changes_sign_when_direction_is_reversed():
    a=directional_state([3.],[1.],0.1).iloc[0].comm_directionality
    b=directional_state([1.],[3.],0.1).iloc[0].comm_directionality
    assert np.isclose(a,-b)
