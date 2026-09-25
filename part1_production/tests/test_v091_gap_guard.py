import numpy as np
from strata_hierarchy.v091.completion import build_backbone
def test_long_knn_gap_is_not_forced_when_local_scale_is_small():
    xy=np.array([[0.,0.],[.1,0.],[.2,0.],[10.,0.],[10.1,0.],[10.2,0.]])
    b=build_backbone(xy,np.empty((0,2),int),k=2,gap_factor=2.0,local_k=1)
    assert not (((b.cell_i_index<=2)&(b.cell_j_index>=3)).any())
