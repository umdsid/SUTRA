import numpy as np
from strata_hierarchy.v1092.reconstruct import membership_map_from_label_vector,partition_invariants
def test_label_vector_is_exact_partition():
    m=membership_map_from_label_vector(np.array([0,0,1,2]))
    r=partition_invariants(m)
    assert r["nodes"]==3 and r["unique"]==4 and r["disjoint"]
