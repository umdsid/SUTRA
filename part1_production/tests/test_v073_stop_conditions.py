from strata_hierarchy.v073.full import natural_stop_reason


def test_no_admissible_is_natural_stop():
    assert natural_stop_reason(10,0,0)=="no_admissible_boundaries"


def test_available_selected_does_not_stop():
    assert natural_stop_reason(10,3,2) is None


def test_no_boundaries_is_natural_stop():
    assert natural_stop_reason(0,0,0)=="no_remaining_supernode_boundaries"
