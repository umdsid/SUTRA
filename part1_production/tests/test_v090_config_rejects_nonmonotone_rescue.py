import pytest
from strata_hierarchy.v090.slow_flow import SlowFlowConfig

def test_rescue_schedule_must_expand():
    with pytest.raises(ValueError):
        SlowFlowConfig(
            epsilon_fraction=.001,
            rescue_fraction_1=.0005
        ).validate()
