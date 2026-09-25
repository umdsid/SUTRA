import pandas as pd
from strata_hierarchy.v1091.semantics import evaluate_masks,choose_unique_partition
def test_explicit_active_mask_can_recover_partition():
    d=pd.DataFrame({
      "node_id":["a","b","old"],
      "members":[["1","2"],["3"],["1"]],
      "is_active":[True,True,False]
    })
    e=evaluate_masks(d,2,3)
    c=choose_unique_partition(e)
    assert c is not None and c["candidate"]=="is_active=true"
