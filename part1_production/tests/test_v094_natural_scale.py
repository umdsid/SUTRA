import numpy as np
from strata_hierarchy.v094.diagnostics import natural_scale
def test_natural_scale_is_log_ratio():
    x=natural_scale(100,np.array([100,50,25]))
    assert np.allclose(x,[0,np.log(2),np.log(4)])
