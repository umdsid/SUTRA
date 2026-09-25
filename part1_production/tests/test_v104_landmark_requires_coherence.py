import pandas as pd
from strata_hierarchy.v104.basins import hierarchy_landmarks
def test_low_coherence_candidate_is_not_landmark():
    b=pd.DataFrame([{"candidate_landmark":2,"basin_landmarks":5,"lifetime_ell":.2,
       "basin_depth":1.,"minimum_removed_fraction":.4,"minimum_nodes":100}])
    c=pd.DataFrame([{"candidate_landmark":2,"cross_block_coherence":.3,
       "unstable_block_switch_entropy":1.,"stable_blocks":3,"total_blocks":10,"active_blocks":"x"}])
    cfg={"min_basin_landmarks":3,"min_cross_block_coherence":.5}
    x=hierarchy_landmarks(b,c,cfg)
    assert not bool(x.iloc[0].hierarchy_landmark)
