from strata_hierarchy.v078.holonomy import reverse_loop_same_basepoint

def test_inverse_loop_keeps_basepoint():
    assert reverse_loop_same_basepoint((0,1,2))==(0,2,1)
    assert reverse_loop_same_basepoint((4,7,9,12))==(4,12,9,7)
