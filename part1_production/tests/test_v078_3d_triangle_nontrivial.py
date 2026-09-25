import numpy as np
from strata_hierarchy.v078.holonomy import compose_loop_holonomy,holonomy_metrics

def test_three_dimensional_direction_triangle_can_have_nontrivial_holonomy():
    Q=np.array([
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.],
    ])
    H,U=compose_loop_holonomy(Q,(0,1,2))
    m=holonomy_metrics(H)
    assert m["orthogonality_error"]<1e-12
    assert m["spectral_angle_max"]>1e-3
