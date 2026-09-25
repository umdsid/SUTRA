import numpy as np
from strata_hierarchy.v094.diagnostics import contiguous_true_runs
def test_multiple_stationary_regimes_are_retained():
    m=np.array([0,1,1,0,1,1,1,0],bool)
    assert contiguous_true_runs(m)==[(1,2),(4,6)]
