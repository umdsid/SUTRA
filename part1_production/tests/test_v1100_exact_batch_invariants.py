import numpy as np,pandas as pd
from strata_hierarchy.v1100.invariants import assert_batch_exact
def test_two_disjoint_merges_are_exact():
    before=np.array([0,1,2,3,4])
    b=pd.DataFrame({"super_i":[0,2],"super_j":[1,3]})
    after=np.array([0,0,2,2,4])
    r=assert_batch_exact(before,after,b,5)
    assert r["mass_exact"] and r["nodes_after"]==3
