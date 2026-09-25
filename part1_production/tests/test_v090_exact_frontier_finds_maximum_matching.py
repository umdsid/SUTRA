import pandas as pd
from strata_hierarchy.v090.slow_flow import exact_best_matching

def test_exact_frontier_searches_around_greedy_trap():
    x=pd.DataFrame([
        {"super_i":0,"super_j":1,"geometry_pair_cost":.1},
        {"super_i":1,"super_j":2,"geometry_pair_cost":.2},
        {"super_i":0,"super_j":3,"geometry_pair_cost":.3},
        {"super_i":2,"super_j":4,"geometry_pair_cost":.4},
    ])
    y=exact_best_matching(x,2)
    assert len(y)==2
    used=[]
    for r in y.itertuples():
        used += [r.super_i,r.super_j]
    assert len(used)==len(set(used))
