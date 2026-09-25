import numpy as np
from strata.geometry_benchmark.baselines import adjacency_from_labels

def test_adjacency_from_labels():
    x=np.array([[1,1,2,2],[1,1,2,2]],dtype=int)
    assert adjacency_from_labels(x)=={(1,2)}
