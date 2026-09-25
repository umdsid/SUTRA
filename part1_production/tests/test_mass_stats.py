
import numpy as np
from strata_brain_publication.core import mass_stats
def test_mass():
    s=mass_stats(np.array([0,0,1,2,2,2]))
    assert s["n_nodes"]==3 and s["total_mass"]==6
    assert 0 < s["n_eff_fraction"] <= 1
