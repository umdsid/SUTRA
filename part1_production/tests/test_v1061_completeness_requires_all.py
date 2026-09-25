import pandas as pd
from strata_hierarchy.v1061.resolve import sample_complete
def test_missing_group_holds():
    p=pd.DataFrame([
      {"landmark_id":"L1","group":"expression","resolved":True},
      {"landmark_id":"L1","group":"functional","resolved":False},
    ])
    s=pd.DataFrame([{"node_state_available":True}])
    c={"required_observable_groups":["expression","functional"],"require_node_state":True}
    assert not sample_complete(p,s,c)
