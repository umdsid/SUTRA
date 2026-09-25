import numpy as np
from strata_hierarchy.v077.transport import canonical_metric_sqrt,apply_transport

def test_near_antipodal_full_transport_is_stable():
    G=np.array([[2.0,.25],[.25,1.4]])
    H,Hinv,w=canonical_metric_sqrt(G)
    qi=np.array([.4,0.])
    theta=np.pi-2e-5
    qj=.3*np.array([np.cos(theta),np.sin(theta)])
    v=np.array([.37,-.81])
    z=apply_transport(H,Hinv,qi,qj,v)
    zz=apply_transport(H,Hinv,qj,qi,z)
    assert np.isclose(v@(G@v),z@(G@z),rtol=2e-11,atol=2e-11)
    assert np.allclose(zz,v,rtol=2e-10,atol=2e-10)
