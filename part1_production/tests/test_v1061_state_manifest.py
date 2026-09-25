import pandas as pd
from strata_hierarchy.v1061.resolve import state_path_for_candidate
def test_v104_manifest_path_is_reused():
    d=pd.DataFrame({"candidate_landmark":[7],"node_state_available":[True],
                    "node_state_path":["/tmp/x.parquet"]})
    assert str(state_path_for_candidate(d,7))=="/tmp/x.parquet"
