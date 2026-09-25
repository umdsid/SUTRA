import numpy as np
from sutra.hierarchy.v078.holonomy import reverse_inverse_error

def test_reversed_loop_is_inverse_at_same_basepoint():
    Q=np.array([
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.],
    ])
    assert reverse_inverse_error(Q,(0,1,2))<1e-12
