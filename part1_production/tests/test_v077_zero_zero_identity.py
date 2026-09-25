import numpy as np
from sutra.hierarchy.v077.transport import classify_pair

def test_zero_zero_is_symmetric_identity():
    z=np.zeros(3)
    assert classify_pair(z,z)["transport_status"]=="IDENTITY_SYMMETRIC"
