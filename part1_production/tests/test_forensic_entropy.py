from strata_native_mechanics.nullspace_forensics import _entropy
def test_entropy():
    assert _entropy([0,0,0],bins=4,period=1.0)==0.0
    assert _entropy([0,.25,.5,.75],bins=4,period=1.0)>0
