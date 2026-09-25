import pandas as pd, numpy as np
from strata_hierarchy.v094.diagnostics import identify_stationary_windows
def test_two_node_tail_not_eligible():
    d=pd.DataFrame({
        "nodes":[100,80,60,40,20,10,3,2],
        "removed_fraction":[0,.2,.4,.6,.8,.9,.97,.98],
        "ell":np.arange(8,dtype=float)
    })
    v=np.ones(8)*.1; a=np.ones(8)*.1
    w,m,vt,at=identify_stationary_windows(
        d,v,a,min_nodes=20,min_removed=.05,max_removed=.995,
        velocity_quantile=1.,acceleration_quantile=1.,min_landmarks=2
    )
    assert not m[-1] and not m[-2] and not m[-3]
