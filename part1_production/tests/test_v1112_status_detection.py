
import pandas as pd
from strata_hierarchy.v1112.recovery import support_col
def test_status_detection_rejects_continuous():
    d=pd.DataFrame({"mass_gini":[.1,.2],"mass_status":["PASS","HOLD"]})
    assert support_col(d,"mass",["mass_status"])=="mass_status"
