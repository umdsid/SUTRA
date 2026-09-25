import numpy as np
import pandas as pd
from sutra.hierarchy.v072.pilot import communication_relations


def test_reciprocity_detects_symmetric_pair():
    Y=np.array([[2.,1.],[2.,1.]],dtype=np.float32)
    E=pd.DataFrame({"cell_i_index":[0],"cell_j_index":[1]})
    R=pd.DataFrame([{
        "sender_gene":"A","receiver_gene":"B","kind":"contact",
        "panel_supported":True,
    }])
    C=communication_relations(Y,E,("A","B"),R)
    assert np.isclose(C.iloc[0].comm_reciprocity,1.0)
