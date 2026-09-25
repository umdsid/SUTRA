import numpy as np
import pandas as pd
from strata.mechanics.regularized_solver import build_tension_coupling

def test_coupling_created_for_shared_junction():
    e=pd.DataFrame([
        {"junction0":0,"junction1":1,"tangent_x":1.0,"tangent_y":0.0},
        {"junction0":0,"junction1":2,"tangent_x":0.8,"tangent_y":0.6},
        {"junction0":3,"junction1":4,"tangent_x":0.0,"tangent_y":1.0},
    ])
    L=build_tension_coupling(e)
    assert L.shape[1]==3
    assert L.shape[0]>=1

def test_coupling_zero_without_shared_junction():
    e=pd.DataFrame([
        {"junction0":0,"junction1":1,"tangent_x":1.0,"tangent_y":0.0},
        {"junction0":2,"junction1":3,"tangent_x":0.0,"tangent_y":1.0},
    ])
    L=build_tension_coupling(e)
    assert L.shape[0]==0
