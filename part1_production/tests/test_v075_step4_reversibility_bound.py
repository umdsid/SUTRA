import numpy as np
from strata_hierarchy.v075.directional_norm import (
    certify_one_covector,theoretical_reversal_bound
)

def test_worst_direction_saturates_reversibility_bound():
    G=np.eye(2)
    b=np.array([.5,0.])
    c=certify_one_covector(G,b)
    assert np.isclose(c["worst_direction_ratio"],3.0)
    assert np.isclose(theoretical_reversal_bound(.5),3.0)
