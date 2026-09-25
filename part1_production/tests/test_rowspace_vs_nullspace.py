import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from scipy.linalg import null_space
from strata_native_mechanics.rowspace_observability import (
    RowspaceConfig,patch_rowspace_observability
)

def test_qr_matches_svd_random_rank_deficient():
    rng=np.random.default_rng(17)
    B=rng.normal(size=(8,5))
    # rank <= 5 matrix with 8 variables
    A=csr_matrix(B@rng.normal(size=(5,8)))
    meta={"eidx":{i:i for i in range(8)},"cidx":{},"bidx":{}}
    E=pd.DataFrame(columns=["interface_id","kind","cell_i","cell_j"])
    sm,V,C=patch_rowspace_observability(
        A,meta,E,RowspaceConfig(rank_rcond=1e-9,observable_tol=1e-8)
    )
    Z=null_space(A.toarray(),rcond=1e-9)
    old=np.linalg.norm(Z,axis=1)<=1e-8
    new=(V.sort_values("column_index").rowspace_status.to_numpy()=="OBSERVABLE")
    assert np.array_equal(old,new)
