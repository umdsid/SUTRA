from strata_hierarchy.v075.metric_core import canonical_feature_order,feature_order_sha256
def test_feature_hash_order_sensitive():
    a=canonical_feature_order(["A","B","C"])
    b=canonical_feature_order(["A","B","C"])
    c=canonical_feature_order(["B","A","C"])
    assert feature_order_sha256(a)==feature_order_sha256(b)
    assert feature_order_sha256(a)!=feature_order_sha256(c)
    assert a[-2:]==("__tension_z__","__delta_pressure_z__")
