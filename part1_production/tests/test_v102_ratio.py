import numpy as np
from strata_hierarchy.v102.future_blind import ratio
def test_missing_or_zero_history_is_unresolved():
    assert np.isnan(ratio(1.,0.)) and np.isnan(ratio(1.,np.nan)) and ratio(1.,2.)==.5
