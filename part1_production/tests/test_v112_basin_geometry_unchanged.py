
import pandas as pd
from strata_hierarchy.v112.final_landscape import recompute_basin_supports
def test_geometry_unchanged():
    b=pd.DataFrame({"center_eval":[2],"center_nodes":[80],"entry_eval":[1],"exit_eval":[3],
                    "lifetime":[3],"depth":[1.2],"block_coherence":[.8]})
    t=pd.DataFrame({"eval":[0,1,2,3],"mass_support_exact":[0,1,1,0],
                    "expression_support_exact":[0,1,1,1],
                    "spatial_support_exact":[0,0,1,1]})
    o=recompute_basin_supports(b,t)
    for c in ["center_eval","center_nodes","entry_eval","exit_eval","lifetime","depth","block_coherence"]:
        assert o.loc[0,c]==b.loc[0,c]
