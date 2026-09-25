import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from strata_native_mechanics.identifiability_v2 import BoundaryReductionConfig, reduce_boundary
def test_small_shared():
    A=csr_matrix(np.eye(5))
    meta={"eidx":{"e":0},"cidx":{"c":1},"bidx":{10:2,11:3,12:4}}
    bg=pd.DataFrame([
        {"background_component":10,"kind":"exterior","pixels":1000},
        {"background_component":11,"kind":"internal_gap","pixels":20},
        {"background_component":12,"kind":"internal_gap","pixels":30},
    ])
    R,m=reduce_boundary(A,meta,bg,"small_gap_shared",BoundaryReductionConfig(64,256))
    assert m["n_boundary_variables_reduced"]==2
    assert R.shape[1]==4
