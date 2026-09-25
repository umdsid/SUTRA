import pandas as pd
from strata_hierarchy.v111.audit import pareto
def test_pareto():
 c={'pareto_objectives':{'maximize':['lifetime','depth','block_coherence','expression_support','spatial_support']}};d=pd.DataFrame([dict(lifetime=4,depth=1,block_coherence=.5,expression_support=1,spatial_support=1),dict(lifetime=3,depth=.5,block_coherence=.4,expression_support=1,spatial_support=1),dict(lifetime=2,depth=2,block_coherence=.8,expression_support=.5,spatial_support=1)]);o=pareto(d,c);assert list(o.pareto_layer)==[1,2,1]
