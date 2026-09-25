import numpy as np,pandas as pd
from strata_hierarchy.v091.completion import spatial_expression_weight
def test_weight_defined_without_threshold_veto():
    e=pd.DataFrame({"spatial_distance":[1.,2.],"local_scale_i":[1.,1.],"local_scale_j":[1.,1.]})
    w,ds,de,s=spatial_expression_weight(e,np.array([.9,.1]))
    assert np.isfinite(w).all() and (w>0).all()
