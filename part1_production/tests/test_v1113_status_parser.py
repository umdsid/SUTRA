
from strata_hierarchy.v1113.replay import _extract_status_triplet
def test_triplet():
    x=_extract_status_triplet({"mass_status":"PASS","expression_status":"HOLD","spatial_status":"PASS"})
    assert x=={"mass":"PASS","expr":"HOLD","spatial":"PASS"}
