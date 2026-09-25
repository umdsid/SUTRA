from strata_hierarchy.v090.slow_flow import SlowFlowConfig,flow_landmark_due

def test_landmark_fires_on_reduction_increment():
    c=SlowFlowConfig(heavy_landmark_reduction_fraction=.0025)
    due,x=flow_landmark_due(.0051,.0025,c)
    assert due
    assert abs(x-.005)<1e-12
