import numpy as np,pandas as pd
from strata_hierarchy.v091.completion import functional_distance_edges
def test_identical_expression_has_zero_functional_distance():
    Y=np.array([[1.,2.],[1.,2.]],dtype=np.float32)
    e=pd.DataFrame({"cell_i_index":[0],"cell_j_index":[1]})
    d=functional_distance_edges(Y,e,np.eye(2))
    assert abs(d[0])<1e-12
