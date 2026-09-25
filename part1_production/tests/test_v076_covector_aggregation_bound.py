import numpy as np
from scipy import sparse
from strata_hierarchy.v076.geometry_hierarchy import aggregate_covectors

def test_supernode_covector_is_mean_not_sum():
    B=sparse.csr_matrix(np.array([[1.,0.],[0.,1.]]))
    labels=np.array([0,0])
    out=aggregate_covectors(B,labels,np.array([0])).toarray()
    assert np.allclose(out,[[.5,.5]])
