import numpy as np, pandas as pd
from strata_hierarchy.v105.pareto_audit import pareto_front
def test_missing_pareto_metric_is_ineligible():
    x=pd.DataFrame({"lifetime_ell":[1.,2.],"basin_depth":[1.,np.nan],
                    "cross_block_coherence":[.5,.8]})
    y=pareto_front(x,["lifetime_ell","basin_depth","cross_block_coherence"])
    assert not bool(y.loc[1,"pareto_eligible"])
