import pytest
from strata_hierarchy.v075.bound_calibration import global_scale
def test_target_must_be_strictly_less_than_one():
    with pytest.raises(ValueError):
        global_scale(1.0,1.0)
