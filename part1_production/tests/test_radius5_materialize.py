from strata.production_geometry.materialize_radius5 import adjacency_from_labels
import numpy as np
def test_adjacency():
    x=np.array([[1,1,2],[1,3,2]],dtype=np.int32)
    e=adjacency_from_labels(x)
    assert (1,2) in e and (1,3) in e and (2,3) in e
