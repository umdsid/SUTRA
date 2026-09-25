from strata_hierarchy.v107.domain_audit import gini
def test_equal_mass_gini_zero():
    assert abs(gini([1,1,1]))<1e-12
