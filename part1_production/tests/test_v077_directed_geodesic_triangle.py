import numpy as np,pandas as pd
from strata_hierarchy.v077.geodesics import build_directed_cost_graph,triangle_audit

def test_dijkstra_geodesics_satisfy_triangle_inequality():
    e=pd.DataFrame({
        "super_i":[0,1,0],
        "super_j":[1,2,2],
        "geometry_state_resolved":[True,True,True],
        "local_forward_cost":[1.,1.,3.],
        "local_reverse_cost":[1.2,1.1,2.8],
    })
    A,_=build_directed_cost_graph(e,3)
    t=triangle_audit(A,np.array([0,1,2]))
    assert t["pass"]
