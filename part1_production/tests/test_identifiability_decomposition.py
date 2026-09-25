import numpy as np
from scipy.sparse import csr_matrix
from strata_native_mechanics.identifiability_v2 import nullity_attribution
def test_attr():
    A=csr_matrix(np.array([[1.,1.,0.],[0.,0.,0.]]))
    meta={"eidx":{"e":0},"cidx":{"c":1},"bidx":{"b":2}}
    base,attr=nullity_attribution(A,meta)
    assert base["structural_nullity"]==2
    assert attr["boundary_pressure"]["nullity_removed_if_class_removed"]>=1
