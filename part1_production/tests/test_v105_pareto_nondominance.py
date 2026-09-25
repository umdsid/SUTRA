import pandas as pd
from strata_hierarchy.v105.pareto_audit import pareto_front
def test_dominated_basin_is_removed():
    x=pd.DataFrame({
      "candidate_landmark":[1,2,3],
      "lifetime_ell":[1.,2.,1.5],
      "basin_depth":[1.,2.,3.],
      "cross_block_coherence":[.5,.6,.4]
    })
    y=pareto_front(x,["lifetime_ell","basin_depth","cross_block_coherence"])
    assert not bool(y.loc[0,"pareto_front"])
    assert bool(y.loc[1,"pareto_front"])
    assert bool(y.loc[2,"pareto_front"])
