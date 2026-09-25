import numpy as np
from strata_hierarchy.v101.replay import finite_rms

def test_all_nan_row_stays_nan_and_partial_row_is_finite():
    A=np.array([[np.nan,np.nan],[3.,4.]])
    y=finite_rms(A)
    assert np.isnan(y[0])
    assert np.isclose(y[1],np.sqrt((9+16)/2))
