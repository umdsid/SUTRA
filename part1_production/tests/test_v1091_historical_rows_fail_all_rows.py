import pandas as pd
from strata_hierarchy.v1091.semantics import evaluate_masks
def test_all_rows_not_disjoint_when_history_present():
    d=pd.DataFrame({"node_id":["a","b","old"],"members":[["1"],["2"],["1","2"]],"is_active":[True,True,False]})
    e=evaluate_masks(d,2,2)
    r=e[e.candidate=="all_rows"].iloc[0]
    assert not r.exact_partition
