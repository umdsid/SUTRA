import pandas as pd
from strata_hierarchy.v106.materialize import compute_lsi
def test_lsi_is_descriptive():
    x=pd.DataFrame({"lifetime_ell":[1.,2.],"basin_depth":[2.,1.],"cross_block_coherence":[.2,.4],"pareto_layer":[1,1]})
    y=compute_lsi(x,{})
    assert not y.lsi_is_gate.any()
