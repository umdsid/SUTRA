import numpy as np
from strata_hierarchy.v075.directional_norm import directional_norm,alpha

def test_reversal_identity():
    G=np.diag([1.,2.,3.])
    b=np.array([.1,.05,0.])
    v=np.array([1.,-.4,.3])
    fp=directional_norm(G,b,v)
    fm=directional_norm(G,b,-v)
    assert np.isclose(fp+fm,2*alpha(G,v))
