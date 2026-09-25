import pandas as pd
from strata_hierarchy.v074.effective_state import maximal_matching


def test_effective_matching_is_disjoint():
    x=pd.DataFrame([
        {"super_i":0,"super_j":1,"admissible":True,"ordering_merit":3.,
         "n_boundary_edges":1},
        {"super_i":1,"super_j":2,"admissible":True,"ordering_merit":2.,
         "n_boundary_edges":1},
        {"super_i":3,"super_j":4,"admissible":True,"ordering_merit":1.,
         "n_boundary_edges":1},
    ])
    y=maximal_matching(x)
    z=y[y.selected]
    used=[]
    for r in z.itertuples():
        used += [r.super_i,r.super_j]
    assert len(used)==len(set(used))
