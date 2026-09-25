from strata_hierarchy.v109.audit import compare_partitions
def test_orphan_child_detected():
    child={"a":{"1","2"}}
    parent={"p":{"1"}}
    d,s=compare_partitions(child,parent)
    assert s["children_orphaned"]==1
