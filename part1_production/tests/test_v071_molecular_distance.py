import numpy as np
import pandas as pd
from strata_hierarchy.v071.block_preflight import molecular_edge_distance


def test_molecular_distance_identity_metric():
    Y=np.array([[1.,0.],[0.,1.]],dtype=np.float32)
    E=pd.DataFrame({"cell_i_index":[0],"cell_j_index":[1]})
    d=molecular_edge_distance(Y,E,np.zeros((2,2)),lam=1.0)
    assert np.allclose(d,[np.sqrt(2)])
