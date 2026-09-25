import pandas as pd
from strata_hierarchy.v106.materialize import assign_landmark_ids,validate_order
def test_ids_follow_scale_order():
    x=pd.DataFrame({"minimum_removed_fraction":[.6,.2,.4],"pareto_layer":[1,1,1]})
    y=assign_landmark_ids(x,"s")
    assert list(y.landmark_id)==["s:L1","s:L2","s:L3"]
    assert validate_order(y)
