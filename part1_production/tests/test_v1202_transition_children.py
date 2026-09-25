import numpy as np
from strata_hierarchy.v120.atlas import component_transition
def test_children():
 m=component_transition(np.array([0,0,1,1,2,2]),np.array([0,0,0,0,1,1])); assert set(m[0])=={0,1}
