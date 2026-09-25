import pandas as pd
from strata_hierarchy.v074.directional_comm import support_floor_sensitivity


def test_support_sensitivity_is_monotone_in_quantile():
    e=pd.DataFrame({
        "comm_ij":[0.,.1,.2,.3,1.],
        "comm_ji":[0.,.1,.2,.3,1.],
    })
    s=support_floor_sensitivity(e,(0.01,.05,.1,.2))
    assert s.support_floor.is_monotonic_increasing
    assert s.supported_fraction_all_edges.is_monotonic_decreasing
