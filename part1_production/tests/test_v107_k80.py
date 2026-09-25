from strata_hierarchy.v107.domain_audit import cumulative_k
def test_k80_concentrated_mass():
    assert cumulative_k([80,10,5,5],.80)==1
