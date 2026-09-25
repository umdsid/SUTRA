import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from strata_native_mechanics.observability import ObservabilityConfig,patch_observability

def test_pressure_difference_can_be_observable():
    A=csr_matrix(np.array([[0.,1.,-1.],[1.,0.,0.]]))
    meta={"eidx":{0:0},"cidx":{1:1,2:2},"bidx":{}}
    E=pd.DataFrame([{"interface_id":0,"kind":"cell_cell","cell_i":1,"cell_j":2}])
    sm,v,c=patch_observability(
        A,meta,E,ObservabilityConfig(dense_nvar_cap=10,dense_nullity_cap=10)
    )
    assert len(c)==1
    assert bool(c.iloc[0].numerically_observable)
