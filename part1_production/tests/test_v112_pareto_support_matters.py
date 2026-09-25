
import pandas as pd
from strata_hierarchy.v112.final_landscape import pareto_layer1
def test_support_can_dominate():
    d=pd.DataFrame([
      {"lifetime":5,"depth":2,"block_coherence":.8,"expression_support":1,"spatial_support":1},
      {"lifetime":4,"depth":1,"block_coherence":.7,"expression_support":.5,"spatial_support":.5},
    ])
    o=pareto_layer1(d,["lifetime","depth","block_coherence","expression_support","spatial_support"])
    assert o["pareto_layer"].tolist()==[1,2]
