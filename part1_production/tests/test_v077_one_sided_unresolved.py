import numpy as np
from sutra.hierarchy.v077.transport import classify_pair

def test_one_sided_direction_is_not_fabricated():
    a=np.array([1.,0.])
    z=np.zeros(2)
    assert classify_pair(a,z)["transport_status"]=="ONE_SIDED_UNRESOLVED"
