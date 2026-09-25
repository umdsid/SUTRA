import numpy as np
from strata_hierarchy.v075.bound_calibration import global_scale,scaled_dual_norms
def test_global_scaling_preserves_ratios():
    raw=np.array([.1,.2,.4])
    k=global_scale(.4,.9)
    x=scaled_dual_norms(raw,k)
    assert np.isclose(x[1]/x[0],2.)
    assert np.isclose(x[2]/x[1],2.)
