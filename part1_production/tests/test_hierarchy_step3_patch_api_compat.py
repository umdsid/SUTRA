import pytest

from strata_hierarchy.builders.patches import build_patches
from strata_hierarchy.identifiers import CellID, GraphID, InterfaceID


def test_legacy_detached_cell_ids_alias_still_works():
    p=build_patches(
        [{1,2}],
        {1:"a",2:"b"},
        {"a":CellID(0),"b":CellID(1),"x":CellID(2)},
        {7:0},
        GraphID(0),
        detached_cell_ids=(CellID(2),),
    )
    q=[x for x in p if int(x.id)==-1][0]
    assert tuple(map(int,q.cell_ids))==(2,)
    assert q.interface_ids==()


def test_new_retention_api_can_include_geometry_only_interface():
    p=build_patches(
        [],
        {},
        {"x":CellID(2)},
        {},
        GraphID(0),
        retention_cell_ids=(CellID(2),),
        retention_interface_ids=(InterfaceID(17),),
    )
    q=p[0]
    assert int(q.id)==-1
    assert tuple(map(int,q.cell_ids))==(2,)
    assert tuple(map(int,q.interface_ids))==(17,)


def test_old_and_new_cell_arguments_cannot_both_be_used():
    with pytest.raises(ValueError):
        build_patches(
            [],
            {},
            {},
            {},
            GraphID(0),
            retention_cell_ids=(CellID(1),),
            detached_cell_ids=(CellID(2),),
        )
