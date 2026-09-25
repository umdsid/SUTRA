from strata_hierarchy.v107.domain_audit import effective_number_from_mass
def test_equal_masses_have_full_effective_count():
    assert abs(effective_number_from_mass([1,1,1,1])-4)<1e-12
