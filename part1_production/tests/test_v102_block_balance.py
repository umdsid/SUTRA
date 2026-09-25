import numpy as np
from strata_hierarchy.v102.future_blind import block_disp
def test_blocks_have_equal_authority():
    d,r,n=block_disp(np.array([1.,1.,3.]),np.zeros(3),["A","A","B"],.5,1)
    assert np.isclose(d,np.sqrt((1+9)/2)) and r==2 and n==2
