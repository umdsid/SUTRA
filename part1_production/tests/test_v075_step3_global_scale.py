import numpy as np
from strata_hierarchy.v075.bound_calibration import global_scale,scaled_dual_norms
def test_global_scaling_hits_target_at_global_max():
    raw=np.array([0.,.2,1.001])
    k=global_scale(1.001,.9)
    out=scaled_dual_norms(raw,k)
    assert np.isclose(out.max(),.9)
