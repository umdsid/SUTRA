import numpy as np
from scipy import sparse
from strata_hierarchy.v091.completion import harmonic_fill
def test_harmonic_completion_preserves_anchor_exactly():
    A=sparse.csr_matrix(np.array([[0,1,0],[1,0,1],[0,1,0]],float))
    m=np.array([True,False,True]); v=np.array([2.,0.,4.])
    x,c,p=harmonic_fill(A,m,v,max_iter=1000,tol=1e-12)
    assert x[0]==2. and x[2]==4.
    assert abs(x[1]-3.)<1e-8
