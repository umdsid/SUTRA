import numpy as np
from strata_hierarchy.v077.transport import canonical_metric_sqrt

def test_symmetric_whitening_is_permutation_equivariant():
    G=np.array([[2.0,.4,.1],[.4,1.8,.2],[.1,.2,1.1]])
    P=np.eye(3)[[2,0,1]]
    H,Hinv,w=canonical_metric_sqrt(G)
    Hp,Hpinv,wp=canonical_metric_sqrt(P@G@P.T)
    assert np.allclose(Hp,P@H@P.T,rtol=2e-12,atol=2e-12)
    assert np.allclose(Hpinv,P@Hinv@P.T,rtol=2e-12,atol=2e-12)
