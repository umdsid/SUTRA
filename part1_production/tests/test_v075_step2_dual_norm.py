import numpy as np
from scipy import sparse
from strata_hierarchy.v075.directional_covector import dual_norms_spd

def test_dual_norm_matches_identity_case():
    G=np.eye(3)
    B=sparse.csr_matrix(np.array([[3.,4.,0.],[0.,0.,0.]]))
    d=dual_norms_spd(G,B)
    assert np.allclose(d,[5.,0.])
