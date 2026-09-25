import numpy as np
from scipy import sparse
from strata_hierarchy.v075.metric_core import build_symmetric_base_metric,alpha_norm
def test_alpha_positive_homogeneous():
    L=sparse.csr_matrix(np.array([[1.,-1.],[-1.,1.]]))
    G=build_symmetric_base_metric(L,.5,2)
    v=np.array([1.,2.,.3,-.2])
    for c in (.1,2.,7.):
        assert np.isclose(alpha_norm(G,c*v),c*alpha_norm(G,v),rtol=1e-12,atol=1e-12)
