import pandas as pd
from strata_hierarchy.v1093.replay import inspect_merge_schema
def test_common_merge_schema_resolves():
    d=pd.DataFrame({"merge_a":[1],"merge_b":[2],"new_node_id":[3],"microstep":[1]})
    s=inspect_merge_schema(d)
    assert s["left"]=="merge_a" and s["right"]=="merge_b" and s["new"]=="new_node_id"
