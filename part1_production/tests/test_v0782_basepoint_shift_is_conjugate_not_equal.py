import numpy as np
from strata_hierarchy.v078.holonomy import compose_loop_holonomy

def test_shifted_reverse_need_not_equal_same_basepoint_inverse():
    Q=np.array([
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.],
    ])
    H,U=compose_loop_holonomy(Q,(0,1,2))
    # Plain reversal changes basepoint to node 2.
    Hshift,_=compose_loop_holonomy(Q,(2,1,0),U=U)
    # This is generally not the inverse matrix at basepoint 0.
    assert np.linalg.norm(Hshift@H-np.eye(3),ord="fro")>1e-3
