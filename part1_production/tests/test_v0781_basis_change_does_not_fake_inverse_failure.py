import numpy as np
from strata_hierarchy.v078.holonomy import (
    compose_loop_holonomy,
    reverse_loop_same_basepoint,
)

def test_same_operator_in_changed_basis_requires_conjugation():
    Q=np.array([
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.],
    ])
    loop=(0,1,2)
    H,U=compose_loop_holonomy(Q,loop)

    C=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    U2=U@C
    inv=reverse_loop_same_basepoint(loop)
    Hr2,_=compose_loop_holonomy(Q,inv,U=U2)

    # U2=U C => M_U = C M_U2 C^T.
    Hr_in_U=C@Hr2@C.T
    assert np.linalg.norm(Hr_in_U@H-np.eye(3),ord="fro") < 1e-11
