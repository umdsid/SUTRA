import numpy as np
from strata_hierarchy.v108.mass_spectrum import normalized_mass
def test_normalized_mass_sums_one():
    p=normalized_mass([1,2,3])
    assert abs(np.nansum(p)-1)<1e-12
