from strata_hierarchy.builders.patches import build_patches
from strata_hierarchy.identifiers import CellID, GraphID


def test_detached_cells_get_retention_patch_without_interfaces():
    patches=build_patches(
        [{1,2}],
        {1:"a",2:"b"},
        {"a":CellID(0),"b":CellID(1),"x":CellID(2)},
        {7:0},
        GraphID(0),
        detached_cell_ids=(CellID(2),),
    )
    assert len(patches)==2
    detached=[p for p in patches if int(p.id)==-1][0]
    assert tuple(map(int,detached.cell_ids))==(2,)
    assert detached.interface_ids==()
