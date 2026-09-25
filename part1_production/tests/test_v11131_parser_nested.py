
from strata_hierarchy.v1113.replay import _extract_status_triplet
def test_nested_parser():
    x={"gates":{"mass":{"status":"PASS"},"expression":{"pass":False},"spatial":{"stable":True}}}
    assert _extract_status_triplet(x)=={"mass":"PASS","expr":"HOLD","spatial":"PASS"}
