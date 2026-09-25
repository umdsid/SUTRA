import pandas as pd
from strata_hierarchy.v1061.resolve import exact_match_row
def test_exact_landmark_key_wins():
    d=pd.DataFrame({"landmark_index":[3,7],"nodes":[100,50],"expression_x":[1.,2.]})
    r,m=exact_match_row(d,{"candidate_landmark":7,"minimum_nodes":50,
        "minimum_removed_fraction":.5,"minimum_ell":1.})
    assert m=="exact_landmark_index" and r.expression_x==2.
