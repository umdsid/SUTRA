import pandas as pd

from strata_hierarchy.builders.interfaces import build_interfaces
from strata_hierarchy.builders.patches import build_patches
from strata_hierarchy.identifiers import CellID, GraphID, InterfaceID


def test_geometry_only_boundary_interface_is_retained_without_mechanics():
    E=pd.DataFrame([{
        "interface_id":17,
        "kind":"cell_boundary",
        "cell_i":5,
        "cell_j":None,
        "length":2.0,
        "curvature":0.1,
    }])
    out=build_interfaces(
        E,
        {5:"isolated"},
        {"isolated":CellID(0)},
        pd.DataFrame(columns=["interface_id","tension_production"]),
        pd.DataFrame(columns=["interface_id","delta_p_production"]),
        {},{},
    )
    assert len(out)==1
    e=out[0]
    assert int(e.patch_id)==-1
    assert int(e.owner)==0
    assert e.tension is None
    assert e.delta_pressure is None


def test_retention_patch_can_hold_boundary_only_primary_cell_and_interface():
    p=build_patches(
        [],
        {},
        {"isolated":CellID(0)},
        {},
        GraphID(0),
        retention_cell_ids=(CellID(0),),
        retention_interface_ids=(InterfaceID(17),),
    )
    assert len(p)==1
    assert int(p[0].id)==-1
    assert tuple(map(int,p[0].cell_ids))==(0,)
    assert tuple(map(int,p[0].interface_ids))==(17,)
