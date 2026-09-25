import numpy as np
from strata_hierarchy.v077.transport import rotate_minimal

def test_near_antipodal_reverse_is_inverse():
    a=np.array([1.,0.,0.])
    theta=np.pi-5e-5
    b=np.array([np.cos(theta),np.sin(theta),0.])
    x=np.array([.13,-.31,.91])
    y=rotate_minimal(a,b,x)
    z=rotate_minimal(b,a,y)
    assert np.allclose(z,x,rtol=1e-11,atol=1e-11)
