import pandas as pd
from strata_hierarchy.v1061.resolve import exact_match_row
def test_nearby_removed_fraction_is_not_accepted():
    d=pd.DataFrame({"removed_fraction":[.4,.5001],"expression_x":[1.,2.]})
    r,m=exact_match_row(d,{"candidate_landmark":7,"minimum_nodes":50,
        "minimum_removed_fraction":.5,"minimum_ell":1.})
    assert r is None and m is None
