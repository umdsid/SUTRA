import numpy as np
from strata_hierarchy.v078.holonomy import compose_loop_holonomy

def test_optional_basis_does_not_change_forward_matrix():
    Q=np.array([
        [1.,.1,0.],
        [0.,1.,.2],
        [.2,0.,1.],
    ])
    H,U=compose_loop_holonomy(Q,(0,1,2))
    H2,_=compose_loop_holonomy(Q,(0,1,2),U=U)
    assert np.allclose(H,H2,rtol=1e-13,atol=1e-13)
