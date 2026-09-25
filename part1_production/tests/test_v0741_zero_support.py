import numpy as np
from strata_hierarchy.v074.effective_state import (
    communication_support_and_reciprocity,
)


def test_zero_support_is_not_reciprocal():
    support,supported,r=communication_support_and_reciprocity(
        np.array([0.0]),np.array([0.0]),support_floor=0.1
    )
    assert support[0]==0.0
    assert not bool(supported[0])
    assert np.isnan(r[0])
