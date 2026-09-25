import numpy as np,pandas as pd
from strata_hierarchy.v1100.invariants import LineageTracker,assert_lineage_matches_labels
def test_lineage_parent_mass_is_sum():
    t=LineageTracker.level0(4)
    b=pd.DataFrame({"super_i":[0,2],"super_j":[1,3]})
    a=t.apply(b,7)
    labels=np.array([0,0,2,2])
    assert (a.groupby("parent_token").child_mass.sum().to_numpy()==2).all()
    assert_lineage_matches_labels(t,labels,4)
