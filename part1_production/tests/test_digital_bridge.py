import numpy as np
from strata.connectivity.digital_bridge import repair_diagonal_only_labels

def test_single_diagonal_bridge():
    x=np.zeros((5,5),dtype=np.int32)
    x[1,1]=1
    x[2,2]=1
    y,r=repair_diagonal_only_labels(x)
    assert r["pre_disconnected_4"]==1
    assert r["pre_disconnected_8"]==0
    assert r["post_disconnected_4"]==0
    assert r["bridge_pixels_added"]==1

def test_never_overwrites_foreign_labels():
    x=np.zeros((5,5),dtype=np.int32)
    x[1,1]=1; x[2,2]=1
    x[1,2]=2; x[2,1]=3
    y,r=repair_diagonal_only_labels(x)
    assert y[1,2]==2 and y[2,1]==3
    assert r["diagonal_bridge_failures"]==1
