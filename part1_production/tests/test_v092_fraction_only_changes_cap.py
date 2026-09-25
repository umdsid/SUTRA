import numpy as np,pandas as pd
from strata_hierarchy.v092.sweep import run_schedule

def test_fraction_changes_schedule_not_scientific_columns():
    e=pd.DataFrame({
        "cell_i_index":[0,1,2,3],
        "cell_j_index":[1,2,3,4],
        "spatial_expression_weight":[.8,.8,.8,.8],
        "expression_distance":[.1,.1,.1,.1],
        "normalized_spatial_distance":[1.,1.,1.,1.],
        "functional_distance":[.2,.2,.2,.2],
        "functional_similarity":[.8,.8,.8,.8],
        "cellchat_forward":[0.,0.,0.,0.],
        "cellchat_reverse":[0.,0.,0.,0.],
        "cellchat_total":[0.,0.,0.,0.],
        "cellchat_directionality":[0.,0.,0.,0.],
        "cellchat_support_forward":[0.,0.,0.,0.],
        "cellchat_support_reverse":[0.,0.,0.,0.],
        "tension_complete":[0.,0.,0.,0.],
        "tension_confidence":[0.,0.,0.,0.],
        "delta_p_potential_complete":[0.,0.,0.,0.],
        "pressure_confidence":[0.,0.,0.,0.],
        "spatial_distance":[1.,1.,1.,1.],
        "gap_guard_ratio":[1.,1.,1.,1.],
    })
    w=dict(space_expr=1.,functional=1.,communication=1.,tension=1.,pressure=1.)
    _,h1,_=run_schedule(e,5,.2,w,max_steps=2,target_removed=.2)
    _,h2,_=run_schedule(e,5,.4,w,max_steps=2,target_removed=.2)
    assert len(h1)>0 and len(h2)>0
