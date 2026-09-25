import numpy as np
from scipy import sparse
from strata_brain_pipeline.core import component_means
def test_means():
 X=sparse.csr_matrix([[1,0],[3,2],[10,4]],dtype=float);M,m,_,_=component_means(X,np.array([0,0,1]));A=M.toarray();assert np.allclose(A[0],[2,1]);assert np.allclose(A[1],[10,4]);assert list(m)==[2,1]
