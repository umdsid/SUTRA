import pandas as pd
from sutra.hierarchy.v078.holonomy import registry_lookup,loop_status

def test_one_sided_edge_makes_loop_unresolved():
    reg=pd.DataFrame([
        {"super_i":0,"super_j":1,"transport_status":"DIRECTIONAL_ROTATION"},
        {"super_i":1,"super_j":2,"transport_status":"ONE_SIDED_UNRESOLVED"},
        {"super_i":0,"super_j":2,"transport_status":"IDENTITY_SYMMETRIC"},
    ])
    s=loop_status((0,1,2),registry_lookup(reg))
    assert not s["fully_resolved"]
    assert s["n_unresolved_edges"]==1
