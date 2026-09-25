import pandas as pd
from sutra.hierarchy.v072.pilot import PilotConfig,greedy_matching


def test_matching_is_disjoint_and_capped():
    x=pd.DataFrame([
        {"super_i":0,"super_j":1,"admissible":True,"ordering_merit":3.,
         "n_boundary_edges":1},
        {"super_i":1,"super_j":2,"admissible":True,"ordering_merit":2.,
         "n_boundary_edges":1},
        {"super_i":3,"super_j":4,"admissible":True,"ordering_merit":1.,
         "n_boundary_edges":1},
    ])
    cfg=PilotConfig(max_pair_fraction=1.0,max_pairs_per_level=2)
    y=greedy_matching(x,5,cfg)
    z=y[y.selected]
    used=[]
    for r in z.itertuples():
        used += [r.super_i,r.super_j]
    assert len(used)==len(set(used))
    assert len(z)==2
