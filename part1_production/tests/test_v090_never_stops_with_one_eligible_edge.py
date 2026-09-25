import pandas as pd
from strata_hierarchy.v090.slow_flow import SlowFlowConfig,choose_slow_batch

def test_one_eligible_edge_is_always_selected():
    x=pd.DataFrame([{
        "super_i":0,"super_j":1,
        "admissible":True,
        "geometry_state_resolved":True,
        "geometry_pair_cost":1.0,
        "alpha_cost":1.0,
        "directional_correction":0.0,
        "ordering_merit":1.0,
        "n_boundary_edges":1,
    }])
    r=choose_slow_batch(x,100,0,SlowFlowConfig())
    assert len(r["eligible"])==1
    assert len(r["selected"])==1
