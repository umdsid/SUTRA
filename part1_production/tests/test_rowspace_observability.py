import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from strata_native_mechanics.rowspace_observability import (
    RowspaceConfig,rowspace_basis_qr,patch_rowspace_observability
)

def test_identity_all_coordinates_observable():
    A=csr_matrix(np.eye(4))
    Q,m=rowspace_basis_qr(A)
    assert m["numerical_nullity"]==0
    assert np.allclose(Q@Q.T,np.eye(4),atol=1e-10)

def test_one_equation_two_variables_unresolved():
    A=csr_matrix(np.array([[1.,1.]]))
    meta={"eidx":{0:0,1:1},"cidx":{},"bidx":{}}
    E=pd.DataFrame(columns=["interface_id","kind","cell_i","cell_j"])
    sm,V,C=patch_rowspace_observability(A,meta,E,RowspaceConfig())
    assert sm["numerical_nullity"]==1
    assert set(V.rowspace_status)=={"UNRESOLVED"}

def test_pressure_difference_observable_under_common_gauge():
    # p1-p2 is constrained while the common pressure shift is free.
    A=csr_matrix(np.array([[1.,-1.]]))
    meta={"eidx":{},"cidx":{1:0,2:1},"bidx":{}}
    E=pd.DataFrame([{
        "interface_id":7,"kind":"cell_cell","cell_i":1,"cell_j":2
    }])
    sm,V,C=patch_rowspace_observability(A,meta,E,RowspaceConfig())
    assert sm["numerical_nullity"]==1
    assert len(C)==1
    assert C.iloc[0].rowspace_status=="OBSERVABLE"
    assert set(V.rowspace_status)=={"UNRESOLVED"}
