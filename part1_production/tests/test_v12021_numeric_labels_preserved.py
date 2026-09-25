
import numpy as np
from strata_hierarchy.v120.atlas import component_transition
def test_numeric_labels():
    e=np.array([0,0,1,1,2,2])
    c=np.array([0,0,0,0,1,1])
    m=component_transition(e,c)
    assert set(m[0])=={0,1}
