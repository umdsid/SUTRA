import pandas as pd
from strata_hierarchy.v074.directional_comm import (
    directional_state,readiness_audit
)


def test_readiness_requires_bounded_finite_supported_directionality():
    d=directional_state([2.,1.],[1.,2.],0.1)
    s=pd.DataFrame({
        "positive_quantile":[.01,.1],
        "support_floor":[.1,.2],
        "supported_fraction_all_edges":[1.,1.],
        "n_supported":[2,2],
    })
    r=readiness_audit(d,s)
    assert r["metric_readiness_structural"]
    assert r["bounded_directionality"]
