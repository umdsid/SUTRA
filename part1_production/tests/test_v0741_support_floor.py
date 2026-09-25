import pandas as pd
from strata_hierarchy.v074.effective_state import (
    freeze_communication_support_floor,
)


def test_support_floor_ignores_zero_support():
    e=pd.DataFrame({
        "comm_ij":[0.,0.1,1.0],
        "comm_ji":[0.,0.1,1.0],
    })
    d=freeze_communication_support_floor(e,positive_quantile=0.0)
    assert d["communication_support_floor"]>0
    assert d["n_positive_support"]==2
    assert d["n_zero_or_nearzero_support"]==1
