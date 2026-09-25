from strata_hierarchy.v109.audit import compare_partitions
def test_nested_mass_monotone():
    child={"a":{"1","2"},"b":{"3"}}
    parent={"p":{"1","2","3"}}
    d,s=compare_partitions(child,parent)
    assert s["mass_monotone_all"]
    assert s["children_nested"]==2
