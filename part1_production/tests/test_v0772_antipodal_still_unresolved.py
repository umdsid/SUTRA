import numpy as np, pytest
from strata_hierarchy.v077.transport import rotate_minimal,classify_pair

def test_exact_antipodal_policy_is_unchanged():
    a=np.array([1.,0.])
    b=-a
    assert classify_pair(a,b)["transport_status"]=="ANTIPODAL_UNRESOLVED"
    with pytest.raises(ValueError):
        rotate_minimal(a,b,np.array([.2,.3]))
