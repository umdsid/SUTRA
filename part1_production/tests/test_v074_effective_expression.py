import numpy as np
from strata_hierarchy.v074.effective_state import aggregate_supernode_expression


def test_effective_expression_is_mean_of_constituents():
    Y=np.array([[1.,0.],[3.,2.],[5.,4.]],dtype=np.float32)
    labels=np.array([0,0,2])
    ids,S,n=aggregate_supernode_expression(Y,labels)
    assert ids.tolist()==[0,2]
    assert n.tolist()==[2,1]
    assert np.allclose(S[0],[2.,1.])
    assert np.allclose(S[1],[5.,4.])
