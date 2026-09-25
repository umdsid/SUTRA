import numpy as np
from strata_hierarchy.v075.directional_norm import certify_one_covector

def test_dual_bound_violation_is_not_certified():
    G=np.eye(2)
    b=np.array([1.1,0.])
    c=certify_one_covector(G,b)
    assert not c["certificate_pass"]
    assert not c["strict_dual_bound"]
