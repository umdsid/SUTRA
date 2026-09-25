import numpy as np
from strata_hierarchy.v077.transport import rotate_minimal

def test_near_antipodal_rotation_is_norm_preserving():
    # Deliberately close to antipodal but outside the unresolved tolerance.
    a=np.array([1.,0.,0.,0.])
    theta=np.pi-2e-5
    b=np.array([np.cos(theta),np.sin(theta),0.,0.])
    x=np.array([.2,-.7,.4,.1])
    y=rotate_minimal(a,b,x)
    assert np.isclose(np.linalg.norm(y),np.linalg.norm(x),rtol=1e-12,atol=1e-12)
    assert np.allclose(rotate_minimal(a,b,a),b,rtol=1e-10,atol=1e-10)
