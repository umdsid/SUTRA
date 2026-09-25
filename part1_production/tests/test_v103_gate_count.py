import pandas as pd
from strata_hierarchy.v103.audit import gate_table
def test_three_of_four_is_identified():
    f=pd.DataFrame([{"velocity_ratio":.5,"acceleration_ratio":.5,"closure_support":.2,
        "resolved_closure_lags":3,"eligible":True,"closure_median_ratio":.5,
        "landmark_index":1,"nodes":100,"removed_fraction":.2,"active":True,"plateau":False}])
    c={"velocity_ratio_max":.72,"acceleration_ratio_max":.82,
       "closure_support_required":.67,"min_resolved_closure_lags":2,"closure_ratio_max":.78}
    g=gate_table(f,c).iloc[0]
    assert g.gate_count==3 and g.near_plateau_3of4
