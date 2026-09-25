import numpy as np
from strata.connectivity.raster_repair import remove_detached_inferred_fragments,audit_disconnected

def test_inferred_fragment_removed_not_refilled():
    o=np.zeros((12,12),np.int32);o[4:7,4:7]=1
    x=o.copy();x[1,1]=1
    y,n,p=remove_detached_inferred_fragments(x,o)
    assert n==1 and p==1
    assert y[1,1]==0
    assert audit_disconnected(y)==[]
