import numpy as np
from strata_hierarchy.v075.bound_calibration import certify_scaled_norms
def test_strict_bound_certification():
    c=certify_scaled_norms(np.array([0.,.1,.9]),.9)
    assert c["strict_unit_bound"]
    assert c["certificate_pass"]
    assert np.isclose(c["minimum_finsler_positivity_margin"],.1)
