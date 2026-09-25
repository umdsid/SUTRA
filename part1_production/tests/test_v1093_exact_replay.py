import numpy as np,pandas as pd
from strata_hierarchy.v1093.replay import replay_merge_table
def test_replay_hits_requested_counts():
    initial=np.arange(5)
    d=pd.DataFrame({"left":[0,2,3],"right":[1,3,4],"new":[5,6,7],"step":[1,2,3]})
    r=replay_merge_table(d,initial,[4,3,2])
    assert set(r["snapshots"])=={4,3,2}
    assert r["applied_merges"]==3
