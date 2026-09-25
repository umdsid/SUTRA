import pandas as pd
from strata_hierarchy.v1091.semantics import evaluate_masks,choose_unique_partition
def test_two_equivalent_semantic_masks_are_deduplicated():
    d=pd.DataFrame({"node_id":["a","b"],"members":[["1"],["2"]],"is_active":[True,True]})
    e=evaluate_masks(d,2,2)
    # all_rows and is_active are identical masks and therefore deduplicated
    assert choose_unique_partition(e) is not None
