import pandas as pd
from strata_hierarchy.v073.full import maximal_disjoint_matching


def test_maximal_matching_has_no_pilot_cap():
    rows=[]
    for k in range(100):
        rows.append({
            "super_i":2*k,
            "super_j":2*k+1,
            "admissible":True,
            "ordering_merit":float(100-k),
            "n_boundary_edges":1,
        })
    x=pd.DataFrame(rows)
    y=maximal_disjoint_matching(x)
    assert int(y.selected.sum())==100


def test_competing_edges_remain_disjoint():
    x=pd.DataFrame([
        {"super_i":0,"super_j":1,"admissible":True,"ordering_merit":3.,
         "n_boundary_edges":1},
        {"super_i":1,"super_j":2,"admissible":True,"ordering_merit":2.,
         "n_boundary_edges":1},
        {"super_i":3,"super_j":4,"admissible":True,"ordering_merit":1.,
         "n_boundary_edges":1},
    ])
    y=maximal_disjoint_matching(x)
    z=y[y.selected]
    used=[]
    for r in z.itertuples():
        used += [int(r.super_i),int(r.super_j)]
    assert len(used)==len(set(used))
    assert int(z.selected.sum())==2
