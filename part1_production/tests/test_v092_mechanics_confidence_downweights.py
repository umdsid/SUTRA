import pandas as pd
from strata_hierarchy.v092.sweep import completed_pair_cost

def test_zero_confidence_removes_mechanics_authority():
    x=pd.DataFrame({
        "expression_distance":[.1],
        "normalized_spatial_distance":[1.],
        "functional_distance":[.2],
        "cellchat_directionality":[0.],
        "cellchat_support_forward":[0.],
        "cellchat_support_reverse":[0.],
        "tension_complete":[1e9],
        "tension_confidence":[0.],
        "delta_p_potential_complete":[1e9],
        "pressure_confidence":[0.],
    })
    w=dict(space_expr=1.,functional=1.,communication=1.,tension=1.,pressure=1.)
    y=completed_pair_cost(x,w)
    assert y.loc[0,"cost_tension"]==0.
    assert y.loc[0,"cost_pressure"]==0.
