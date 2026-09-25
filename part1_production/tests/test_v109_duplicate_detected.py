from strata_hierarchy.v109.audit import partition_audit
def test_duplicate_membership_fails():
    x={"a":{"1","2"},"b":{"2","3"}}
    r=partition_audit(x)
    assert not r["partition_is_disjoint"]
    assert r["duplicate_assignments"]==1
