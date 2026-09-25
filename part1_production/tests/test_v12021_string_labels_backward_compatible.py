
import numpy as np
from strata_hierarchy.v120.atlas import component_transition
def test_string_labels():
    e=np.array(["a","a","b","b","c"])
    c=np.array(["x","x","x","x","y"])
    m=component_transition(e,c)
    assert set(m["x"])=={"a","b"}
