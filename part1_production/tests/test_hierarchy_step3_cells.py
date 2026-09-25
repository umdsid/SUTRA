import pandas as pd

from sutra.hierarchy.builders.cells import build_cells


def test_all_measured_cells_are_retained_in_matrix_order():
    barcodes=["c2","c1","detached"]
    df=pd.DataFrame({
        "cell_id":["c1","c2","detached"],
        "x_centroid":[1.,2.,3.],
        "y_centroid":[4.,5.,6.],
        "cell_area":[10.,11.,12.],
    })
    cells,m=build_cells(
        barcodes,df,
        {"c1":0,"c2":0},
        {"c1":(7,), "c2":(8,)},
    )
    assert [int(c.id) for c in cells]==[0,1,2]
    assert [c.gene_index for c in cells]==[0,1,2]
    assert int(cells[2].patch_id)==-1
    assert cells[2].interface_ids==()
