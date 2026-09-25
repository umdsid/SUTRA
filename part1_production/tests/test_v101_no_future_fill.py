import numpy as np
from strata_hierarchy.v101.replay import causal_difference

def test_future_value_does_not_resolve_missing_past_endpoint():
    Y=np.array([[np.nan],[1.],[3.]])
    x=np.array([0.,1.,2.])
    D=causal_difference(Y,x,1)
    assert np.isnan(D[1,0])
    assert np.isclose(D[2,0],2.)
