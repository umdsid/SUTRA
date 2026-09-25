import pandas as pd
from strata.vmsi.export_roi import choose_connected_roi


def test_chooses_largest_component():
    rows=[{"cell_i":f"L{i}","cell_j":f"L{i+1}"} for i in range(9)]
    rows += [{"cell_i":"S0","cell_j":"S1"},{"cell_i":"S1","cell_j":"S2"}]
    e=pd.DataFrame(rows)
    ids=[f"L{i}" for i in range(10)]+["S0","S1","S2"]
    c=pd.DataFrame({
        "cell_id":ids,
        "x_centroid":[100+i for i in range(10)]+[0,1,2],
        "y_centroid":[0]*13,
    })
    out=choose_connected_roi(e,c,5)
    assert len(out)==5
    assert all(v.startswith("L") for v in out)
