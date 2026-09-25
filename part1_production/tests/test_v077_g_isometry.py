import numpy as np

from strata_hierarchy.v077.transport import (
    canonical_metric_sqrt,
    apply_transport,
)


def test_transport_preserves_alpha():
    G=np.diag([1.,2.,3.])
    H,Hinv,_=canonical_metric_sqrt(G)
    qi=np.array([.4,0.,0.])
    qj=np.array([0.,.2,0.])
    v=np.array([.3,-.7,.2])

    w=apply_transport(H,Hinv,qi,qj,v)

    assert np.isclose(
        v@(G@v),
        w@(G@w),
        rtol=1e-11,
        atol=1e-11,
    )
