from strata_hierarchy.v1092.reconstruct import compare_nested
def test_parent_mass_never_smaller():
    c={"0":{0,1}}
    p={"a":{0,1,2}}
    d,ok=compare_nested(c,p)
    assert ok and bool(d.mass_monotone.iloc[0])
