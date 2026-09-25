import numpy as np
from strata_hierarchy.v1100.metrics import mass_statistics
def test_effective_domains_and_k80():
    r=mass_statistics(np.array([80,10,10]))
    assert abs(r["effective_domain_number"]-(1/(.8**2+.1**2+.1**2)))<1e-12
    assert r["K80"]==1
