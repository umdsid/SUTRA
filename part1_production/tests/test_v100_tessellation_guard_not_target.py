from strata_hierarchy.v100.adaptive import RegimeDetector
def test_nodes_below_guard_cannot_create_stable_regime():
    cfg={"min_effective_nodes":20,"min_removed_fraction":.1,"max_removed_fraction":.995,
         "stable_support_required":.6,"persistence_landmarks":2,"confirmation_landmarks":2}
    d=RegimeDetector(cfg)
    c={"stable_support":1.0,"median_velocity":.1}
    assert d.update(1,c,10,.9)=="CONTINUE"
    assert d.low_run==0
