import numpy as np
from strata.connectivity.repair import audit_and_repair
def test_inferred_fragment():
    o=np.zeros((12,12),np.int32); o[4:7,4:7]=1
    x=o.copy(); x[1,1]=1
    y,r=audit_and_repair(x,o)
    assert r["status"]=="PASS"
    assert r["postrepair_disconnected_labels"]==0
