import numpy as np
from sutra.hierarchy.v076.pressure_tail_audit import robust_tail_threshold
def test_tail_threshold_is_above_q999():
    x=np.r_[np.linspace(0,1,10000),1e6]
    t=robust_tail_threshold(x)
    assert t["robust_tail_threshold"]>=t["q999_abs"]
    assert t["max_abs"]>100*t["q999_abs"]
