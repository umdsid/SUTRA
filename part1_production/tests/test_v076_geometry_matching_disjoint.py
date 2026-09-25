import pandas as pd
from sutra.hierarchy.v076.geometry_hierarchy import geometry_matching

def test_geometry_matching_is_disjoint():
    x=pd.DataFrame([
        {"super_i":0,"super_j":1,"admissible":True,"geometry_state_resolved":True,"geometry_pair_cost":1.,"alpha_cost":1.,"n_boundary_edges":1},
        {"super_i":1,"super_j":2,"admissible":True,"geometry_state_resolved":True,"geometry_pair_cost":1.1,"alpha_cost":1.1,"n_boundary_edges":1},
        {"super_i":3,"super_j":4,"admissible":True,"geometry_state_resolved":True,"geometry_pair_cost":1.2,"alpha_cost":1.2,"n_boundary_edges":1},
    ])
    y=geometry_matching(x)
    z=y[y.selected_geometry]
    used=[]
    for r in z.itertuples(): used += [r.super_i,r.super_j]
    assert len(used)==len(set(used))
    assert len(z)==2
