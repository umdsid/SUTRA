
import numpy as np
from strata_brain_publication.core import subsample_labels
def test_subsample():
    x=np.arange(100)
    q=subsample_labels(x,.8,np.random.default_rng(2))
    assert len(q)==80
