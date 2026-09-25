import pandas as pd
from strata.connectivity.forensics import classify_cell

def test_invalid_polygon_classified():
    g=pd.DataFrame({
        "cell_id":["A"]*4,
        "vertex_x":[0,1,0,1],
        "vertex_y":[0,1,1,0],
        "vertex_order":[0,1,2,3],
    })
    r=classify_cell(g)
    assert r["raw_valid"] is False
