from strata_hierarchy.v1091.semantics import compare_nested
def test_exact_nested_partition():
    d,ok=compare_nested({"a":{"1"},"b":{"2"}},{"p":{"1","2"}})
    assert ok
