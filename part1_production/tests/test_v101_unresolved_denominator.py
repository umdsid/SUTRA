import pandas as pd
from strata_hierarchy.v101.replay import classify_landmarks

def test_unresolved_configs_do_not_vote_against_stability():
    configs=pd.DataFrame([
        {"landmark_index":1,"nodes":100,"removed_fraction":.2,"ell":.2,
         "stride":1,"width":3,"velocity":1.,"acceleration":1.,
         "resolved_blocks":8,"expected_blocks":10,"resolved":True},
        {"landmark_index":1,"nodes":100,"removed_fraction":.2,"ell":.2,
         "stride":2,"width":3,"velocity":None,"acceleration":None,
         "resolved_blocks":3,"expected_blocks":10,"resolved":False},
    ])
    th=pd.DataFrame([
        {"stride":1,"width":3,"calibrated":True,"velocity_low":2.,
         "acceleration_low":2.,"velocity_high":10.,"acceleration_high":10.},
        {"stride":2,"width":3,"calibrated":True,"velocity_low":2.,
         "acceleration_low":2.,"velocity_high":10.,"acceleration_high":10.},
    ])
    d=pd.DataFrame({"landmark_index":[1],"nodes":[100],"removed_fraction":[.2],"ell":[.2]})
    cfg={"min_resolved_configs_absolute":1,"min_resolved_config_fraction":.5,
         "high_support_required":.5,"stable_support_required":.6}
    y=classify_landmarks(configs,th,d,cfg).iloc[0]
    assert y.resolved_configs==1
    assert y.stable_support==1.0
    assert y.state=="LOW_FLOW"
