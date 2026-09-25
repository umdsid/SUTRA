import numpy as np
import pandas as pd
from strata.mechanics.cached_regularization import build_cached_system

def test_operator_normalization_finite():
    e=pd.DataFrame([
        {"cell_i":"A","cell_j":"B","interface_length":1.0,
         "tangent_x":1.0,"tangent_y":0.0,
         "normal_i_to_j_x":0.0,"normal_i_to_j_y":1.0,
         "junction0":0,"junction1":1},
        {"cell_i":"B","cell_j":"C","interface_length":1.0,
         "tangent_x":0.8,"tangent_y":0.6,
         "normal_i_to_j_x":-0.6,"normal_i_to_j_y":0.8,
         "junction0":1,"junction1":2},
    ])
    c=build_cached_system(e,["A","B","C"],3)
    assert np.isfinite(c.coupling_normalization)
    assert c.coupling_normalization>0
