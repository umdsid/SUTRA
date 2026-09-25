import numpy as np
from strata_hierarchy.v078.holonomy import (
    compose_loop_holonomy,
    reverse_loop_same_basepoint,
    reverse_inverse_error,
)

def test_reverse_loop_uses_same_reduced_coordinates_and_basepoint():
    Q=np.zeros((3,8))
    Q[0,[0,3,5]]=[1.0,.2,.1]
    Q[1,[1,3,6]]=[.9,.2,.1]
    Q[2,[2,4,7]]=[1.1,.15,.08]
    Q/=np.linalg.norm(Q,axis=1,keepdims=True)

    loop=(0,1,2)
    H,U=compose_loop_holonomy(Q,loop)
    inv=reverse_loop_same_basepoint(loop)
    Hr,_=compose_loop_holonomy(Q,inv,U=U)

    assert np.linalg.norm(Hr@H-np.eye(H.shape[0]),ord="fro") < 1e-11
    assert reverse_inverse_error(Q,loop) < 1e-11
