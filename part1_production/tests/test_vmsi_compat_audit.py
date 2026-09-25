import numpy as np
from strata.vmsi_compat.audit import disconnected

def test_4_vs_8_connectivity():
    x=np.zeros((5,5),dtype=np.int32)
    x[1,1]=1
    x[2,2]=1
    assert disconnected(x,1)==[1]
    assert disconnected(x,2)==[]
