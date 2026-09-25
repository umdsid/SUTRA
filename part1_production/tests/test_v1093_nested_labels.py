import numpy as np
from strata_hierarchy.v1093.replay import nested_labels
def test_nested_partition_is_certified():
    fine=np.array([0,0,1,2])
    coarse=np.array([0,0,0,1])
    d,ok=nested_labels(fine,coarse)
    assert ok
