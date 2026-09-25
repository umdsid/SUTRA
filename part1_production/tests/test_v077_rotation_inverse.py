import numpy as np
from strata_hierarchy.v077.transport import rotate_minimal

def test_reverse_rotation_is_inverse():
    a=np.array([1.,0.,0.])
    b=np.array([0.,1.,0.])
    x=np.array([.2,.3,.7])
    y=rotate_minimal(a,b,x)
    z=rotate_minimal(b,a,y)
    assert np.allclose(z,x,atol=1e-12)
