import numpy as np
from strata_brain_pipeline.core import sample_actual_pairs
def test_pairs():
 p=np.array([0,0,1,2,2,2]);q=sample_actual_pairs(p,50,100,np.random.default_rng(1));assert (0,1) in q;assert all(p[i]==p[j] for i,j in q)
