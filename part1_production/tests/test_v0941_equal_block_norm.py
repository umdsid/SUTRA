import numpy as np
from strata_hierarchy.v0941.audit import balanced_norm
def test_equal_blocks_not_equal_observables():
    b={"A":np.array([1.]),"B":np.array([3.])}
    assert np.allclose(balanced_norm(b),np.sqrt((1+9)/2))
