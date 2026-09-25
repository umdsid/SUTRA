import numpy as np
from strata_hierarchy.v077.transport import rotate_minimal

def test_minimal_rotation_maps_a_to_b():
    a=np.array([1.,0.,0.])
    b=np.array([0.,1.,0.])
    y=rotate_minimal(a,b,a)
    assert np.allclose(y,b,atol=1e-12)
