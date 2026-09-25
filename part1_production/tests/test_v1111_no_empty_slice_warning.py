
import numpy as np, pandas as pd, warnings
from strata_hierarchy.v1111.audit import activity_state
def test_no_empty_slice_warning():
    d=pd.DataFrame({"speed_a":[np.nan,1.0],"speed_b":[np.nan,2.0]})
    cfg={"block_speed_prefixes":["speed_"]}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a,n,c,cs=activity_state(d,cfg)
    assert len(w)==0
    assert np.isnan(a[0]) and n[0]==0
