import numpy as np
from strata_hierarchy.v075.directional_norm import certify_one_covector

def test_strict_positivity_under_dual_bound():
    G=np.eye(2)
    b=np.array([.6,0.])
    c=certify_one_covector(G,b)
    assert c["certificate_pass"]
    assert c["positive_forward"]
    assert c["positive_reverse"]
