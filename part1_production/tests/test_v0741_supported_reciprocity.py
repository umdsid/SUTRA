import numpy as np
from strata_hierarchy.v074.effective_state import (
    communication_support_and_reciprocity,
)


def test_supported_symmetric_signal_is_reciprocal():
    _,supported,r=communication_support_and_reciprocity(
        np.array([2.0]),np.array([2.0]),support_floor=0.1
    )
    assert bool(supported[0])
    assert np.isclose(r[0],1.0)


def test_supported_one_way_signal_is_asymmetric():
    _,supported,r=communication_support_and_reciprocity(
        np.array([2.0]),np.array([0.0]),support_floor=0.1
    )
    assert bool(supported[0])
    assert r[0] < 1e-9
