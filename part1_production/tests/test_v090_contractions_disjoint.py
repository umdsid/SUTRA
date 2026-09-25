import numpy as np,pandas as pd
from strata_hierarchy.v090.slow_flow import apply_contractions

def test_disjoint_batch_contracts_to_lower_id_survivors():
    labels=np.array([0,1,2,3,4])
    x=pd.DataFrame([
        {"super_i":0,"super_j":1},
        {"super_i":3,"super_j":4},
    ])
    y=apply_contractions(labels,x)
    assert np.array_equal(y,np.array([0,0,2,3,3]))
