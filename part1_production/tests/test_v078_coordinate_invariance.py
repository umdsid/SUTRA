import numpy as np
from sutra.hierarchy.v078.holonomy import coordinate_invariance_check

def test_holonomy_observables_invariant_under_orthogonal_relabeling():
    Q=np.array([
        [1.,0.,0.,0.],
        [0.,1.,0.,0.],
        [0.,0.,1.,0.],
    ])
    assert coordinate_invariance_check(Q,(0,1,2),seed=1)<1e-10
