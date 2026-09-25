import numpy as np,pandas as pd
from strata_hierarchy.v075.directional_covector import build_raw_node_covectors

def test_directional_covector_does_not_invent_mechanics_components():
    Y=np.array([[4.,1.],[1.,4.]])
    E=pd.DataFrame({"cell_i_index":[0],"cell_j_index":[1],"comm_ij":[2.],"comm_ji":[1.]})
    R=pd.DataFrame([{"sender_gene":"A","receiver_gene":"B","kind":"contact","panel_supported":True}])
    b,_,_=build_raw_node_covectors(Y,E,("A","B"),R,0.1)
    assert np.allclose(b.toarray()[:,-2:],0.)
