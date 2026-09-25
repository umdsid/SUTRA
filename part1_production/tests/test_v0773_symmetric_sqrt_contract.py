import numpy as np
from strata_hierarchy.v077.transport import canonical_metric_sqrt

def test_symmetric_square_root_contract():
    G=np.array([[2.0,.3],[.3,1.2]])
    H,Hinv,w=canonical_metric_sqrt(G)
    assert np.allclose(H,H.T,atol=1e-13)
    assert np.allclose(H@H,G,rtol=1e-12,atol=1e-12)
    assert np.allclose(Hinv@H,np.eye(2),rtol=1e-12,atol=1e-12)
