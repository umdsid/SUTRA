import numpy as np
from strata_hierarchy.v10931.replay_shards import nested_labels
def test_nested_labels():
    _,ok=nested_labels(np.array([0,0,1,2]),np.array([0,0,0,1]))
    assert ok
