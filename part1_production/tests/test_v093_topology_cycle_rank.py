import numpy as np,pandas as pd
from strata_hierarchy.v093.flow import topology_stats
def test_triangle_has_one_cycle():
    e=pd.DataFrame({"super_i":[0,1,0],"super_j":[1,2,2]})
    s=topology_stats(e,np.array([0,1,2]))
    assert s["components"]==1
    assert s["cycle_rank"]==1
