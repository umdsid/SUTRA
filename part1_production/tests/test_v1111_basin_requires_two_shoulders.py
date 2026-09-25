
import pandas as pd, numpy as np
from strata_hierarchy.v1111.audit import detect_local_basins
def test_open_tail_not_called_basin():
    # monotone tail has no right enclosing maximum
    n=12
    d=pd.DataFrame({"eval":range(n),"nodes":range(100,88,-1),
                    "mass":[1]*n,"expr":[1]*n,"spatial":[1]*n,
                    "speed_a":[4,3,2,1,2,3,4,3,2,1,.5,.2],
                    "speed_b":[4,3,2,1,2,3,4,3,2,1,.5,.2]})
    cfg={"block_speed_prefixes":["speed_"],"smoothing_window":1,
         "local_extremum_radius":1,"minimum_basin_lifetime_evals":3}
    b=detect_local_basins(d,cfg)
    assert not ((b["exit_eval"]==n-1).any() if len(b) else False)
