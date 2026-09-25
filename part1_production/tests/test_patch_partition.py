import pandas as pd
from strata_native_mechanics.patches import partition_core_cells,add_halo

def test_partition():
    E=pd.DataFrame([
        {"kind":"cell_cell","cell_i":1,"cell_j":2},
        {"kind":"cell_cell","cell_i":2,"cell_j":3},
        {"kind":"cell_cell","cell_i":3,"cell_j":4},
    ])
    p=partition_core_cells(E,patch_size=2)
    assert sum(len(x) for x in p)==4
    h=add_halo(p[0],E,hops=1)
    assert len(h)>=len(p[0])
