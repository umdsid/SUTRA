import numpy as np
from scipy import sparse
from strata_hierarchy.v075.metric_core import build_symmetric_base_metric,alpha_norm
def test_symmetric_base_reversal_symmetry():
    G=build_symmetric_base_metric(sparse.eye(3),1.0,3)
    v=np.array([1.,-.5,.2,.3,-.1])
    assert np.isclose(alpha_norm(G,v),alpha_norm(G,-v))
