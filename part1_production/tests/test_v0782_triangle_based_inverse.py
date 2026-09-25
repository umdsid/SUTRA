import numpy as np
from strata_hierarchy.v078.holonomy import (
    compose_loop_holonomy,
    reverse_loop_same_basepoint,
    reverse_inverse_error,
)

def test_orthogonal_triangle_inverse_at_same_basepoint():
    Q=np.array([
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.],
    ])
    loop=(0,1,2)
    H,U=compose_loop_holonomy(Q,loop)
    inv=reverse_loop_same_basepoint(loop)
    Hr,_=compose_loop_holonomy(Q,inv,U=U)
    assert np.linalg.norm(Hr@H-np.eye(3),ord="fro")<1e-12
    assert reverse_inverse_error(Q,loop)<1e-12
