import pandas as pd
from strata_hierarchy.v109.audit import dominant_by_mass
def test_k80_selection():
    d=pd.DataFrame({"supernode_id":["a","b","c"],"mass":[80,10,10]})
    x=dominant_by_mass(d,.8)
    assert int(x.dominant_K.sum())==1
