import numpy as np
from strata.gap_audit.core import internal_gap_metrics

def test_internal_hole_detected():
    m=np.zeros((20,20),bool)
    m[3:17,3:17]=True
    m[8:12,8:12]=False
    s,filled,internal=internal_gap_metrics(m)
    assert s["n_internal_gap_components"]==1
    assert internal.sum()==16

def test_no_hole():
    m=np.zeros((20,20),bool)
    m[3:17,3:17]=True
    s,filled,internal=internal_gap_metrics(m)
    assert s["n_internal_gap_components"]==0
    assert internal.sum()==0
