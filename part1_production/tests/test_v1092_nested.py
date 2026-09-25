from strata_hierarchy.v1092.reconstruct import compare_nested
def test_nested_labels():
    c={"0":{0,1},"1":{2},"2":{3}}
    p={"a":{0,1,2},"b":{3}}
    d,ok=compare_nested(c,p)
    assert ok
