import pandas as pd
from strata_hierarchy.v106.materialize import build_hierarchy_tree
def test_tree_is_ordered_chain():
    x=pd.DataFrame({"landmark_id":["s:L1","s:L2"],"landmark_order":[1,2],
                    "minimum_removed_fraction":[.2,.5],"principal_landmark":[True,True]})
    t=build_hierarchy_tree(x,"s")
    assert t.iloc[1].parent_id=="s:L0"
    assert t.iloc[2].parent_id=="s:L1"
