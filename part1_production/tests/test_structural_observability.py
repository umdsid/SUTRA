import numpy as np
from scipy.sparse import csr_matrix
from strata_native_mechanics.observability import structural_unresolved_columns

def test_unmatched_variable_propagates():
    A=csr_matrix(np.array([[1.,1.]]))
    u,m=structural_unresolved_columns(A)
    assert u.tolist()==[True,True]
    assert m["unmatched_variables"]==1

def test_identity_is_observable():
    A=csr_matrix(np.eye(4))
    u,m=structural_unresolved_columns(A)
    assert not u.any()
