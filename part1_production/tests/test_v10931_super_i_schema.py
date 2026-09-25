import pandas as pd
from strata_hierarchy.v10931.replay_shards import schema
def test_observed_schema_resolves():
    s=schema(pd.DataFrame({"microstep":[0],"super_i":[1],"super_j":[2]}))
    assert s["left"]=="super_i" and s["right"]=="super_j"
