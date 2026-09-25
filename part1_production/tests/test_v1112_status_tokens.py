
import numpy as np
from strata_hierarchy.v1112.recovery import status_float
def test_tokens():
    assert status_float("PASS")==1
    assert status_float("HOLD")==0
    assert np.isnan(status_float(0.372))
