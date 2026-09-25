import numpy as np
import pandas as pd
from strata_hierarchy.v072.pilot import contract_labels


def test_smaller_label_survives_deterministically():
    labels=np.array([0,1,2,3])
    s=pd.DataFrame({
        "super_i":[0,2],
        "super_j":[1,3],
        "selected":[True,True],
    })
    out,m=contract_labels(labels,s)
    assert out.tolist()==[0,0,2,2]
    assert m=={1:0,3:2}
