
import numpy as np
from strata_brain_publication.core import random_coalescent_masses
def test_coalescent():
    m=random_coalescent_masses(100,17,np.random.default_rng(1))
    assert len(m)==17 and m.sum()==100 and (m>=1).all()
