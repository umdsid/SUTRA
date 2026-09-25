from strata_hierarchy.v100.adaptive import choose_fraction
def test_sharp_change_and_confirmation_are_ultraslow():
    cfg={"fraction_min":.0002,"fraction_base":.0005,"fraction_confirm":.0002}
    a=choose_fraction({"state":"FAST_CHANGE"},cfg,False)[0]
    b=choose_fraction({"state":"REGULAR"},cfg,False)[0]
    c=choose_fraction({"state":"LOW_FLOW"},cfg,False)[0]
    d=choose_fraction({"state":"REGULAR"},cfg,True)[0]
    assert a==.0002 and c==.0002 and d==.0002 and b==.0005
