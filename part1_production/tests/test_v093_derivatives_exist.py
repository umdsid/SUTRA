import pandas as pd
from strata_hierarchy.v093.flow import derivative_table
def test_first_and_second_scale_differences_created():
    d=pd.DataFrame({"removed_fraction":[0.,.1,.2,.3],"x":[0.,1.,4.,9.]})
    y=derivative_table(d)
    assert "d1_x" in y and "d2_x" in y
