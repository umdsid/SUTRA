import numpy as np
from strata_hierarchy.v078.holonomy import reverse_inverse_error

def test_four_cycle_inverse_at_same_basepoint():
    Q=np.array([
        [1.,0.,0.,0.],
        [0.,1.,0.,0.],
        [0.,0.,1.,0.],
        [0.,0.,0.,1.],
    ])
    assert reverse_inverse_error(Q,(0,1,2,3))<1e-12
