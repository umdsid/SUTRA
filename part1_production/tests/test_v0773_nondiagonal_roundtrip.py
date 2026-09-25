import numpy as np
from strata_hierarchy.v077.transport import canonical_metric_sqrt,apply_transport

def test_nondiagonal_roundtrip():
    G=np.array([[2.0,.4,.2],[.4,1.7,.1],[.2,.1,1.3]])
    H,Hinv,w=canonical_metric_sqrt(G)
    qi=np.array([.3,.1,0.])
    qj=np.array([-.1,.2,.15])
    v=np.array([.4,-.7,.2])
    z=apply_transport(H,Hinv,qi,qj,v)
    zz=apply_transport(H,Hinv,qj,qi,z)
    assert np.allclose(zz,v,rtol=5e-12,atol=5e-12)
