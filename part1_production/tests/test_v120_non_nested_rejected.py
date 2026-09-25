
import numpy as np, pytest
from strata_hierarchy.v120.atlas import component_transition
def test_non_nested():
    e=np.array(["a","a","b","b"])
    c=np.array(["x","y","x","x"])
    with pytest.raises(RuntimeError):
        component_transition(e,c)
