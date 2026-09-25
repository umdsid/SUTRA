import numpy as np
from strata.vmsi_compat.audit import upsample_labels_to_subpixel_grid,contingency

def test_subpixel_grid_shape_and_centers():
    x=np.array([[1,2],[3,4]],dtype=np.int32)
    y=upsample_labels_to_subpixel_grid(x)
    assert y.shape==(3,3)
    assert y[0,0]==1 and y[0,2]==2 and y[2,0]==3 and y[2,2]==4
    assert y[1,1]==0

def test_contingency_same_shape():
    x=np.array([[1,0],[0,2]])
    c=contingency(x,x)
    assert c[1][0][0]==1 and c[2][0][0]==2
