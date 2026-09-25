import numpy as np
from strata.production_geometry.materialize import adjacency_from_labels,component_stats

def test_adjacency_and_component():
    lab=np.array([[1,1,2],[1,3,2],[3,3,2]],dtype=np.int32)
    e=adjacency_from_labels(lab)
    assert (1,2) in e
    assert (1,3) in e
    assert (2,3) in e
    c=component_stats(e,3)
    assert c["n_components"]==1
    assert c["largest_component_fraction"]==1.0
