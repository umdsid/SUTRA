import numpy as np
from strata_hierarchy.v075.directional_norm import directional_norm

def test_positive_homogeneity():
    G=np.eye(3)
    b=np.array([.1,-.1,0.])
    v=np.array([1.,2.,3.])
    for c in (.1,2.,5.):
        assert np.isclose(
            directional_norm(G,b,c*v),
            c*directional_norm(G,b,v)
        )
