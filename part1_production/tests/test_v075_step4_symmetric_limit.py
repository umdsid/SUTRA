import numpy as np
from strata_hierarchy.v075.directional_norm import directional_norm

def test_zero_covector_recovers_symmetric_limit():
    G=np.eye(3); b=np.zeros(3); v=np.array([1.,2.,-1.])
    assert np.isclose(directional_norm(G,b,v),directional_norm(G,b,-v))
