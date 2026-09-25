import pandas as pd
from strata_hierarchy.v104.basins import detect_basins
def test_local_minimum_expands_to_finite_basin():
    g=pd.DataFrame({"landmark_index":range(7),"nodes":[100,90,80,70,60,50,40],
      "removed_fraction":[0,.1,.2,.3,.4,.5,.6],"ell":[0,.1,.2,.3,.4,.5,.6],
      "plateau_distance_score":[2,1,.5,.2,.45,1.2,2]})
    c=pd.DataFrame({"landmark_index":[3],"table_index":[3]})
    cfg={"basin_shoulder_multiplier":3.,"basin_absolute_floor":.05,"min_basin_landmarks":3,
         "shoulder_window":1,"basin_overlap_dedup":.65}
    b=detect_basins(g,c,cfg)
    assert len(b)==1 and b.iloc[0].basin_landmarks>=3
