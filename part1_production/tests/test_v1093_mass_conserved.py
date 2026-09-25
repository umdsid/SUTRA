import numpy as np,pandas as pd
from strata_hierarchy.v1093.replay import replay_merge_table,audit_snapshot
def test_every_snapshot_keeps_all_cells():
    initial=np.arange(4)
    d=pd.DataFrame({"left":[0,2],"right":[1,3],"new":[10,11]})
    r=replay_merge_table(d,initial,[3,2])
    for n,lab in r["snapshots"].items():
        q=audit_snapshot(lab,4,n)
        assert q["cell_count_match"] and q["node_count_match"]
