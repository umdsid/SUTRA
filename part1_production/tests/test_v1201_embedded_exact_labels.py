
import numpy as np
from strata_hierarchy.v120.atlas import _array_candidate
def test_embedded():
    a=_array_candidate([0,0,1,1],4,2)
    assert a is not None and len(np.unique(a))==2
