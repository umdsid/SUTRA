import pandas as pd

from sutra.hierarchy.builders.interfaces import build_interfaces
from sutra.hierarchy.identifiers import CellID
from sutra.hierarchy.flags import TENSION_CERTIFIED, DELTA_P_CERTIFIED


def test_only_certified_mechanics_are_attached():
    E=pd.DataFrame([
        {"interface_id":1,"kind":"cell_cell","cell_i":1,"cell_j":2,
         "length":4.,"curvature":.1},
        {"interface_id":2,"kind":"cell_boundary","cell_i":1,"cell_j":None,
         "length":3.,"curvature":.2},
    ])
    T=pd.DataFrame([{"interface_id":1,"tension_production":2.5}])
    D=pd.DataFrame([{"interface_id":1,"delta_p_production":-0.4}])
    out=build_interfaces(
        E,
        {1:"a",2:"b"},
        {"a":CellID(0),"b":CellID(1)},
        T,D,
        {1:0,2:0},
        {1:1,2:1},
    )
    assert out[0].tension==2.5
    assert out[0].delta_pressure==-0.4
    assert out[0].flags & TENSION_CERTIFIED
    assert out[0].flags & DELTA_P_CERTIFIED
    assert out[1].tension is None
    assert out[1].delta_pressure is None
