import numpy as np
from strata_hierarchy.v075.bound_calibration import reversibility_bound_from_rho
def test_reversibility_bound_formula():
    assert np.isclose(reversibility_bound_from_rho(.5),3.)
    assert np.isclose(reversibility_bound_from_rho(.9),19.)
