from strata_hierarchy.v109.audit import partition_audit
def test_disjoint_partition():
    x={"a":{"1","2"},"b":{"3"}}
    r=partition_audit(x)
    assert r["partition_is_disjoint"]
    assert r["unique_level0_members"]==3
