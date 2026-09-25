import pandas as pd
from strata_hierarchy.v102.future_blind import PlateauDetector
def test_quiet_from_birth_cannot_confirm():
    cfg={"entry_history_landmarks":6,"min_active_landmarks_before_entry":2,"persistence_landmarks":3,"confirmation_landmarks":2}
    f=pd.DataFrame([{"landmark_index":i,"plateau":True,"active":False,"velocity_ratio":.2,"acceleration_ratio":.2,"closure_median_ratio":.2} for i in range(5)])
    d=PlateauDetector(cfg); a=[d.update(f,i) for i in range(len(f))]
    assert "ENTER_CONFIRMATION" not in a
