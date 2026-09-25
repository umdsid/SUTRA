
import pandas as pd
from strata_hierarchy.v112.final_landscape import exact_join_support
def test_exact_join():
    t=pd.DataFrame({"eval":[0,1],"nodes":[10,9]})
    r=pd.DataFrame({"evaluation_index":[0,1],"nodes":[10,9],
                    "mass_status":["PASS","HOLD"],
                    "expr_status":["HOLD","PASS"],
                    "spatial_status":["PASS","PASS"]})
    o,c=exact_join_support(t,r)
    assert c=={"mass":1.0,"expr":1.0,"spatial":1.0}
    assert o["mass_support_exact"].tolist()==[1.0,0.0]
