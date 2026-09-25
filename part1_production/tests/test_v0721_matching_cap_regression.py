import pandas as pd

from strata_hierarchy.v072.pilot import PilotConfig, greedy_matching


def test_rejected_overlap_does_not_consume_contraction_capacity():
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
    pairs=set(zip(z.super_i.astype(int),z.super_j.astype(int)))
    assert pairs=={(0,1),(3,4)}
    assert len(z)==2


def test_cap_counts_only_accepted_pairs():
    x=pd.DataFrame([
        {"super_i":0,"super_j":1,"admissible":True,"ordering_merit":5.,
         "n_boundary_edges":1},
        {"super_i":1,"super_j":2,"admissible":True,"ordering_merit":4.,
         "n_boundary_edges":1},
        {"super_i":3,"super_j":4,"admissible":True,"ordering_merit":3.,
         "n_boundary_edges":1},
        {"super_i":5,"super_j":6,"admissible":True,"ordering_merit":2.,
         "n_boundary_edges":1},
    ])
    cfg=PilotConfig(max_pair_fraction=1.0,max_pairs_per_level=2)
    y=greedy_matching(x,7,cfg)
    assert int(y.selected.sum())==2
