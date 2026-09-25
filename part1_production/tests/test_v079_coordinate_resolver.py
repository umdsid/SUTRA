import pandas as pd
from sutra.hierarchy.v079.curvature_summary import resolve_xy_columns

def test_common_centroid_names_resolve():
    x=pd.DataFrame({"centroid_x":[1.],"centroid_y":[2.]})
    assert resolve_xy_columns(x)==("centroid_x","centroid_y")
