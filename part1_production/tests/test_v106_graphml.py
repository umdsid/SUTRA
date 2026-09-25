import pandas as pd
from strata_hierarchy.v106.materialize import graphml_text
def test_graphml_has_chain_edge():
    t=pd.DataFrame([{"node_id":"s:L0","parent_id":None},{"node_id":"s:L1","parent_id":"s:L0"}])
    assert '<edge source="s:L0" target="s:L1"/>' in graphml_text(t)
