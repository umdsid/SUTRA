import numpy as np,pandas as pd
from strata_hierarchy.v075.directional_covector import build_raw_node_covectors

def test_unsupported_edge_adds_no_covector():
    Y=np.array([[4.,1.],[1.,4.]])
    E=pd.DataFrame({"cell_i_index":[0],"cell_j_index":[1],"comm_ij":[0.],"comm_ji":[0.]})
    R=pd.DataFrame([{"sender_gene":"A","receiver_gene":"B","kind":"contact","panel_supported":True}])
    b,s,_=build_raw_node_covectors(Y,E,("A","B"),R,0.1)
    assert b.nnz==0
    assert not s.comm_has_supported_incident_flux.any()
