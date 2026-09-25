import numpy as np
from scipy import sparse
from strata_hierarchy.v075.metric_core import build_symmetric_base_metric,certify_symmetric_base_metric
def test_identity_plus_laplacian_and_positive_mechanics_is_spd():
    L=sparse.csr_matrix(np.array([[1.,-1.],[-1.,1.]]))
    G=build_symmetric_base_metric(L,2.0,2)
    c=certify_symmetric_base_metric(G)
    assert c["positive_definite"] and c["certificate_pass"]
