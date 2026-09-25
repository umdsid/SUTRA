import numpy as np
from strata_hierarchy.v071.block_preflight import robust_location_scale,robust_standardize


def test_robust_scale_and_missingness():
    x=np.array([1.,2.,3.,4.,np.nan])
    s=robust_location_scale(x)
    assert s["scale_valid"]
    z=robust_standardize(x,s)
    assert np.isnan(z[-1])
    assert np.isfinite(z[:-1]).all()
