import numpy as np
from strata_hierarchy.v104.basins import safe_mean,safe_median,safe_quantile
def test_empty_support_is_unresolved_not_warning():
    x=np.array([np.nan,np.nan])
    assert np.isnan(safe_mean(x)) and np.isnan(safe_median(x)) and np.isnan(safe_quantile(x,.5))
