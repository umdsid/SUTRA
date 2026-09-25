from strata_hierarchy.v0941.audit import window_overlap
def test_overlap():
    a={"ell_start":0.,"ell_end":2.}; b={"ell_start":1.,"ell_end":3.}
    assert abs(window_overlap(a,b)-1/3)<1e-12
