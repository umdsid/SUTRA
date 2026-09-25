from strata_hierarchy.v100.adaptive import RegimeDetector
def test_detector_does_not_stop_on_single_quiet_landmark():
    cfg={"min_effective_nodes":20,"min_removed_fraction":.1,"max_removed_fraction":.995,
         "stable_support_required":.6,"persistence_landmarks":3,"confirmation_landmarks":2}
    d=RegimeDetector(cfg)
    c={"stable_support":1.0,"median_velocity":1.0}
    assert d.update(1,c,100,.2)=="CONTINUE"
    assert d.update(2,c,100,.2)=="CONTINUE"
    assert d.update(3,c,100,.2)=="ENTER_CONFIRMATION"
    assert d.update(4,c,100,.2)=="CONFIRMING"
    assert d.update(5,c,100,.2)=="STOP_CONFIRMED_REGIME"
