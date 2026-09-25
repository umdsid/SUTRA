import pandas as pd
from strata_hierarchy.v101.replay import replay_detector

def test_confirmation_is_required():
    rows=[]
    for i in range(8):
        rows.append({"landmark_index":i,"nodes":100,"removed_fraction":.2+i*.01,
                     "ell":.2+i*.01,"eligible_vote":True,"stable_support":1.,
                     "high_support":0.,"median_velocity":1.,"median_acceleration":1.,
                     "resolved_config_fraction":1.,"state":"LOW_FLOW"})
    df=pd.DataFrame(rows)
    cfg={"min_effective_nodes":20,"detector_min_removed_fraction":.1,
         "detector_max_removed_fraction":.995,"stable_support_required":.6,
         "persistence_landmarks":3,"confirmation_landmarks":2}
    tr,stop=replay_detector(df,cfg)
    assert stop is not None
    assert stop["stop_landmark"]==4
