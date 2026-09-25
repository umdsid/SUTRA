from strata_hierarchy.v106.materialize import extract_lineage
def test_missing_state_does_not_create_lineage():
    assert len(extract_lineage(None,"s:L1"))==0
