
import pandas as pd
from strata_hierarchy.v1111.audit import pareto_layer1
def test_pareto():
    cfg={"pareto_objectives":["lifetime","depth","block_coherence","expression_support","spatial_support"]}
    d=pd.DataFrame([
      dict(lifetime=5,depth=2,block_coherence=.8,expression_support=1,spatial_support=1),
      dict(lifetime=4,depth=1,block_coherence=.7,expression_support=1,spatial_support=1),
      dict(lifetime=3,depth=3,block_coherence=.9,expression_support=.5,spatial_support=1)])
    o=pareto_layer1(d,cfg)
    assert o.loc[0,"pareto_layer"]==1
    assert o.loc[1,"pareto_layer"]==2
    assert o.loc[2,"pareto_layer"]==1
